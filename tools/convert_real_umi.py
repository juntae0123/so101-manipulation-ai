#!/usr/bin/env python3
"""Convert normalized real-UMI recordings into the official-UMI four-file input.
정규화된 실 UMI 기록을 공식 UMI 4파일 입력으로 변환한다.

`handoff/umi_adapter/export.py` 가 요구하는 에피소드 구조로 만든다:

    episode_XXXXX/
      metadata.json   {"success": bool, ...}
      recording.npz   timestamp, robot0_eef_pos (n,3),
                      robot0_eef_rot_axis_angle (n,3), robot0_gripper_width (n,1)
      camera0.mp4     손목 RGB
      physics.npz     timestamp, accelerometer, gyroscope, orientation_wxyz

입력 규약은 `SHARE_normalized_umi_source_sample_0916.md` (트랙 A, 2026-09-16).

⚠️ 조용히 틀릴 수 있는 지점이 셋이다. 전부 막았다.

 1. **hand-eye 가 없으면 카메라 포즈가 TCP 포즈가 된다.** 그러면 궤적 전체가
    수십 mm 밀린 채로 학습이 되고, 손실은 잘 떨어지며 실물에서만 실패한다.
    -> `--handeye` 를 필수로 받는다. identity 로 대체하지 않는다.

 2. **ARCore world 는 +Y up 이고 MuJoCo/로봇 쪽은 +Z up 이다.**
    변환을 빼먹으면 "들어올린다" 가 다른 축이 되고, 시뮬 사전학습과 실데이터
    파인튜닝이 서로 다른 중력 방향을 배운다.
    -> `--up-axis` 를 명시적으로 받고 변환 행렬을 기록에 남긴다.

 3. **success 라벨.** 원본에 그 필드가 없다. 자동으로 true 를 만들면
    실패 시연이 학습에 섞인다.
    -> `summary.txt` 의 `verdict` 로 판정하고, 규칙을 metadata 에 적는다.
       판정 불가면 건너뛴다(기본) 또는 `--accept-all` 로 명시적 허용.

    # [로컬] 또는 [서버]
    python convert_real_umi.py --source rec_20260911_145557 \
        --out demos_real/episode_00000 \
        --handeye configs/real/umi_s22_canonical_pinch_side_grasp_provisional.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# ARCore world(+Y up, right-handed) -> robot/sim world(+Z up).
# X 축 기준 +90도. (x, y, z) -> (x, -z, y)
# ⚠️ 이 행렬이 곧 "중력이 어느 축인가" 다. 바꾸면 학습 의미가 달라진다.
R_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]])

ACCEPT_VERDICT_PREFIX = "OK"


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from xyzw quaternions, shape (n,4) -> (n,3,3).
    xyzw 쿼터니언에서 회전행렬. scipy 없이도 돌게 직접 만든다."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-9):
        raise ValueError("영벡터 쿼터니언이 있다")
    x, y, z, w = (q / norm).T
    n = len(x)
    m = np.empty((n, 3, 3), dtype=np.float64)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def matrix_to_rotvec(m: np.ndarray) -> np.ndarray:
    """Rotation matrices -> true axis-angle (rotvec), shape (n,3,3) -> (n,3).
    회전행렬 -> **진짜 axis-angle**. 공식 UMI 입력이 요구하는 표현이다.

    ⚠️ 6D(첫 두 행)가 아니다. 데이터셋 단계의 6D 는 공식 UMI 07 이 내부에서
    만든다. 여기서 6D 를 넣으면 shape 는 맞고 의미만 틀린다."""
    tr = np.clip((m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(tr)
    axis = np.stack([m[:, 2, 1] - m[:, 1, 2],
                     m[:, 0, 2] - m[:, 2, 0],
                     m[:, 1, 0] - m[:, 0, 1]], axis=-1)
    s = np.linalg.norm(axis, axis=-1, keepdims=True)
    out = np.zeros_like(axis)
    # 일반 구간
    ok = (s[:, 0] > 1e-8)
    out[ok] = axis[ok] / s[ok] * angle[ok, None]
    # angle ~ pi 구간: 축을 (R+I) 대각에서 뽑는다
    near_pi = (~ok) & (angle > 1.0)
    for i in np.nonzero(near_pi)[0]:
        a = (m[i] + np.eye(3)) / 2.0
        d = np.clip(np.diag(a), 0.0, None)
        v = np.sqrt(d)
        j = int(np.argmax(v))
        if v[j] < 1e-8:
            continue
        v = v * np.sign(a[j] / v[j])
        out[i] = v / np.linalg.norm(v) * angle[i]
    return out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dict rows, failing loudly on an empty file.
    CSV 를 dict 행으로 읽는다. 비어 있으면 조용히 넘어가지 않고 죽는다."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"행이 없다: {path}")
    return rows


def load_handeye(path: Path) -> tuple[np.ndarray, dict]:
    """Load T_camera_pinch as a 4x4 matrix, plus the raw record for provenance.
    T_camera_pinch 를 4x4 로 읽는다. 원본은 provenance 로 같이 돌려준다.

    받아들이는 형태 — 어느 쪽이든 명시적이어야 한다:
      {"T_camera_pinch": [[4x4]]}
      {"translation_m": [3], "quaternion_xyzw": [4]}
      {"pinch_offset_local_m": [3]}            회전 없음(단위행렬)으로 간주"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    t = np.eye(4)
    if "T_camera_pinch" in raw:
        t = np.asarray(raw["T_camera_pinch"], dtype=np.float64)
        if t.shape != (4, 4):
            raise ValueError(f"T_camera_pinch 가 4x4 가 아니다: {t.shape}")
    elif "translation_m" in raw and "quaternion_xyzw" in raw:
        t[:3, :3] = quat_xyzw_to_matrix(np.asarray([raw["quaternion_xyzw"]]))[0]
        t[:3, 3] = np.asarray(raw["translation_m"], dtype=np.float64)
    elif "pinch_offset_local_m" in raw:
        t[:3, 3] = np.asarray(raw["pinch_offset_local_m"], dtype=np.float64)
    else:
        raise ValueError(
            f"hand-eye 형태를 모르겠다: {sorted(raw)}. "
            "T_camera_pinch / (translation_m + quaternion_xyzw) / "
            "pinch_offset_local_m 중 하나가 있어야 한다")
    return t, raw


def parse_verdict(source: Path) -> tuple[bool | None, str]:
    """Success from summary.txt verdict. None when it cannot be decided.
    summary.txt 의 verdict 로 성공을 판정한다. 못 정하면 None.

    ⚠️ 원본에 success 필드가 없다. 자동으로 true 를 만들면 실패 시연이
    학습에 섞인다. 판정 근거를 문자열로 같이 돌려 metadata 에 남긴다."""
    f = source / "summary.txt"
    if not f.exists():
        return None, "summary.txt 없음"
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith("verdict="):
            v = line.split("=", 1)[1].strip()
            return v.startswith(ACCEPT_VERDICT_PREFIX), f"summary.txt verdict={v}"
    return None, "summary.txt 에 verdict 줄 없음"


def resample_imu(imu_rows: list[dict[str, str]], frame_ts_ns: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, int]:
    """Nearest-sample accel/gyro at each frame timestamp, causal.
    프레임 시각마다 가장 가까운 accel/gyro. **미래값을 쓰지 않는다(causal)**.

    반환: (accel (n,3), gyro (n,3), 허용오차 초과 행 수)"""
    def series(name: str) -> tuple[np.ndarray, np.ndarray]:
        sel = [r for r in imu_rows if r["sensor"] == name]
        if not sel:
            raise ValueError(f"IMU 에 {name} 이 없다")
        ts = np.asarray([int(r["timestamp_ns"]) for r in sel], dtype=np.int64)
        xyz = np.asarray([[float(r["x"]), float(r["y"]), float(r["z"])]
                          for r in sel], dtype=np.float64)
        order = np.argsort(ts)
        return ts[order], xyz[order]

    out = []
    over = 0
    for name in ("accel", "gyro"):
        ts, xyz = series(name)
        # searchsorted 로 "그 시각 이하의 가장 최근 값" 을 고른다
        idx = np.searchsorted(ts, frame_ts_ns, side="right") - 1
        idx = np.clip(idx, 0, len(ts) - 1)
        lag_ms = (frame_ts_ns - ts[idx]) / 1e6
        over += int(np.sum(np.abs(lag_ms) > 20.0))
        out.append(xyz[idx])
    return out[0], out[1], over


def encode_video(source: Path, frame_rows: list[dict[str, str]], dest: Path,
                 fps: float) -> None:
    """Encode ordered JPEGs into camera0.mp4, preserving frame order.
    순서를 보존해 JPEG 를 mp4 로 인코딩한다.

    ffmpeg 가 있으면 그것을, 없으면 OpenCV 를 쓴다. 둘 다 없으면 죽는다 —
    영상 없이 만든 에피소드는 export.py 가 어차피 받지 못한다."""
    listing = dest.parent / f"_frames_{dest.stem}.txt"
    paths = [source / r["image"] for r in frame_rows]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"프레임 파일 없음 {len(missing)}개, 예: {missing[0]}")
    try:
        listing.write_text(
            "".join(f"file '{p.resolve().as_posix()}'\n" for p in paths),
            encoding="utf-8")
        cmd = ["ffmpeg", "-y", "-r", f"{fps:.6f}", "-f", "concat", "-safe", "0",
               "-i", str(listing), "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-loglevel", "error", str(dest)]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and dest.exists():
            return
        err = r.stderr.decode("utf-8", "replace")[-400:]
    except FileNotFoundError:
        err = "ffmpeg 없음"
    finally:
        listing.unlink(missing_ok=True)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(f"ffmpeg 실패({err}) 이고 OpenCV 도 없다") from exc
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise RuntimeError(f"첫 프레임을 못 읽었다: {paths[0]}")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                raise RuntimeError(f"프레임을 못 읽었다: {p}")
            writer.write(img)
    finally:
        writer.release()


def convert(source: Path, out: Path, handeye_path: Path, up_axis: str,
            accept_all: bool) -> int:
    """Convert one recording. Returns 0 on success, non-zero when skipped.
    한 편을 변환한다. 성공 0, 건너뜀은 0 이 아닌 값."""
    poses = read_csv_rows(source / "poses.csv")
    grips = read_csv_rows(source / "gripper.csv")
    frames = read_csv_rows(source / "frames.csv")
    imu = read_csv_rows(source / "imu.csv")
    episode = json.loads((source / "episode.json").read_text(encoding="utf-8"))

    n = len(poses)
    # 세 스트림의 행 수가 다르면 join 이 조용히 어긋난다. 모수를 찍고 죽인다.
    print(f"  행 수  poses {len(poses)} · gripper {len(grips)} · frames {len(frames)}")
    if not (len(grips) == len(frames) == n):
        raise ValueError("poses/gripper/frames 행 수가 다르다")

    ts_ns = np.asarray([int(r["timestamp_ns"]) for r in poses], dtype=np.int64)
    if not np.all(np.diff(ts_ns) > 0):
        raise ValueError("timestamp 가 단조 증가가 아니다 (export.py 가 거부한다)")
    ft_ns = np.asarray([int(r["timestamp_ns"]) for r in frames], dtype=np.int64)
    if not np.array_equal(ts_ns, ft_ns):
        raise ValueError("poses 와 frames 의 timestamp 가 다르다")
    gi = np.asarray([int(r["frame_index"]) for r in grips], dtype=np.int64)
    if not np.array_equal(gi, np.arange(n)):
        raise ValueError("gripper frame_index 가 0..n-1 이 아니다")

    tracking = [r.get("tracking", "") for r in poses]
    bad_track = sum(1 for t in tracking if t != "TRACKING")
    if bad_track:
        print(f"  ⚠️ tracking != TRACKING 인 행 {bad_track}/{n}")

    # --- 포즈: T_world_camera @ T_camera_pinch -> T_world_pinch ---
    quat = np.asarray([[float(r["qx"]), float(r["qy"]), float(r["qz"]),
                        float(r["qw"])] for r in poses], dtype=np.float64)
    pos = np.asarray([[float(r["x"]), float(r["y"]), float(r["z"])]
                      for r in poses], dtype=np.float64)
    t_wc = np.tile(np.eye(4), (n, 1, 1))
    t_wc[:, :3, :3] = quat_xyzw_to_matrix(quat)
    t_wc[:, :3, 3] = pos

    t_cp, handeye_raw = load_handeye(handeye_path)
    t_wp = t_wc @ t_cp

    # --- 좌표계: ARCore(+Y up) -> 로봇/시뮬(+Z up) ---
    if up_axis.upper() == "Y":
        rot = R_YUP_TO_ZUP
        t_wp = t_wp.copy()
        t_wp[:, :3, 3] = t_wp[:, :3, 3] @ rot.T
        t_wp[:, :3, :3] = rot @ t_wp[:, :3, :3]
        up_note = "ARCore +Y up -> +Z up (X축 +90도)"
    elif up_axis.upper() == "Z":
        up_note = "변환 없음 (이미 +Z up)"
    else:
        raise ValueError(f"--up-axis 는 Y 또는 Z 여야 한다: {up_axis}")

    eef_pos = t_wp[:, :3, 3]
    eef_rotvec = matrix_to_rotvec(t_wp[:, :3, :3])
    if not (np.all(np.isfinite(eef_pos)) and np.all(np.isfinite(eef_rotvec))):
        raise ValueError("포즈에 non-finite 값이 있다")

    gap = np.asarray([[float(r["gap_m"])] for r in grips], dtype=np.float64)
    if np.any(gap < 0) or np.any(gap > 0.2):
        raise ValueError(f"gap_m 범위 이상: {gap.min():.4f}~{gap.max():.4f}")

    # --- success: 자동으로 만들지 않는다 ---
    ok, why = parse_verdict(source)
    if ok is None and not accept_all:
        print(f"  !! success 판정 불가 ({why}). 건너뛴다. "
              "강제하려면 --accept-all")
        return 2
    if ok is False and not accept_all:
        print(f"  !! verdict 가 성공이 아니다 ({why}). 건너뛴다")
        return 3
    success = True if accept_all else bool(ok)

    # --- IMU ---
    accel, gyro, over = resample_imu(imu, ts_ns)
    if over:
        print(f"  ⚠️ IMU 지연 20ms 초과 {over}/{n} 행")

    t_sec = (ts_ns - ts_ns[0]) / 1e9
    dur = float(t_sec[-1])
    fps = (n - 1) / dur if dur > 0 else 30.0

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "recording.npz",
        timestamp=t_sec.astype(np.float64),
        robot0_eef_pos=eef_pos.astype(np.float32),
        robot0_eef_rot_axis_angle=eef_rotvec.astype(np.float32),
        robot0_gripper_width=gap.astype(np.float32),
    )
    np.savez_compressed(
        out / "physics.npz",
        timestamp=t_sec.astype(np.float64),
        accelerometer=accel.astype(np.float32),
        gyroscope=gyro.astype(np.float32),
        # ⚠️ 실측이 아니다. 원본에 orientation 이 없고 UmiDataset 이 소비하지도
        # 않는다(export.py 주석). 실측값처럼 기록하지 않기 위해 metadata 에
        # dummy 라고 명시한다.
        orientation_wxyz=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
    )
    encode_video(source, frames, out / "camera0.mp4", fps)

    (out / "metadata.json").write_text(json.dumps({
        "success": success,
        "success_source": why if not accept_all else "--accept-all (강제)",
        "episode_id": episode.get("episode_id"),
        "skill_id": episode.get("skill_id"),
        "source_schema": episode.get("schema"),
        "n_frames": n,
        "duration_s": round(dur, 4),
        "fps_used_for_video": round(fps, 4),
        "pose_chain": "T_world_camera @ T_camera_pinch",
        "rotation_encoding": "true axis-angle (rotvec), radians",
        "world_transform": up_note,
        "handeye_file": handeye_path.name,
        "handeye_raw": handeye_raw,
        "handeye_status": "provisional",
        "orientation_wxyz": "DUMMY — 실측 아님. UmiDataset 미소비",
        "imu_lag_over_20ms_rows": over,
        "tracking_not_ok_rows": bad_track,
        "converter": "convert_real_umi.py/1.0.0",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  OK {out}  ({n}프레임 · {dur:.2f}s · {fps:.1f}Hz · success={success})")
    return 0


def main() -> int:
    """Entry point.
    진입점."""
    ap = argparse.ArgumentParser(description="실 UMI 기록 -> 공식 UMI 4파일 입력")
    ap.add_argument("--source", type=Path, required=True,
                    help="rec_* 디렉터리 하나, 또는 --batch 와 함께 부모 디렉터리")
    ap.add_argument("--out", type=Path, required=True,
                    help="episode_XXXXX 디렉터리, 또는 --batch 와 함께 부모")
    ap.add_argument("--handeye", type=Path, required=True,
                    help="T_camera_pinch. **필수** — 없으면 카메라 포즈가 "
                         "TCP 포즈가 되어 궤적 전체가 밀린다")
    ap.add_argument("--up-axis", default="Y", choices=["Y", "Z", "y", "z"],
                    help="원본 world 의 up 축. ARCore 는 Y")
    ap.add_argument("--batch", action="store_true",
                    help="--source 아래 rec_* 전부를 episode_00000.. 으로")
    ap.add_argument("--accept-all", action="store_true",
                    help="verdict 판정을 무시하고 전부 success=true. "
                         "근거가 metadata 에 그렇게 적힌다")
    args = ap.parse_args()

    if not args.handeye.exists():
        raise SystemExit(f"hand-eye 파일이 없다: {args.handeye}")

    if not args.batch:
        print(f"[1/1] {args.source.name}")
        return convert(args.source, args.out, args.handeye, args.up_axis,
                       args.accept_all)

    srcs = sorted(p for p in args.source.iterdir()
                  if p.is_dir() and p.name.startswith("rec_"))
    if not srcs:
        raise SystemExit(f"rec_* 디렉터리가 없다: {args.source}")
    done = skipped = 0
    for i, s in enumerate(srcs):
        print(f"[{i + 1}/{len(srcs)}] {s.name}")
        try:
            rc = convert(s, args.out / f"episode_{done:05d}", args.handeye,
                         args.up_axis, args.accept_all)
        except Exception as exc:                      # noqa: BLE001
            print(f"  !! 실패: {exc}")
            skipped += 1
            continue
        if rc == 0:
            done += 1
        else:
            skipped += 1
    # 모수를 찍고 합을 검산한다. 조용히 빠진 편이 없어야 한다.
    print(f"\n시도 {len(srcs)} = 변환 {done} + 건너뜀 {skipped}")
    if done + skipped != len(srcs):
        print("!! 합이 안 맞는다. 집계되지 않는 경로가 있다")
        return 4
    if done == 0:
        print("!! 변환된 편이 없다")
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
