"""ARCore bundle -> RawEpisode. **PILOT.** Track A's issue-31 slot stays untouched.
ARCore 번들 -> RawEpisode. **파일럿이다.** 트랙 A 의 이슈 31 칸은 건드리지 않는다.

왜 따로 있는가.

`track_a/convert/arcore.py` 가 정본이고 지금 `NotImplementedError` 다. 그 파일을
우리가 채우면 도경 작업과 섞여서 나중에 **어느 수치가 어느 구현에서 나왔는지 못 되짚는다.**
그래서 우리 것은 여기에 두고 `source` 와 `calibration_id` 에 `pilot` 을 박는다.

두 선행 인자 중 하나는 실측으로 닫혔고, 하나는 **정의로 우회**한다.

**① `T_cam->pinch` — 닫혔다** 🟢 `MEASURE_handeye_from_video_0912.md`
    회전은 HW 도면(중력으로 검정, 3,624프레임 중앙 1.2°),
    평행이동은 영상 실측(5,149프레임, 편별 std <= 2.36mm).

**② `T_ARCore월드->로봇베이스` — 미결(이슈 29). 파일럿은 정의로 우회한다** 🟡
    번들의 월드 원점은 **에피소드마다 다르다**(`world_origin: per_episode_relative`).
    작업대 ArUco 보드가 생기기 전까지는 절대 기준이 없다.

    우회 정의: **에피소드 첫 프레임의 파지점 = 로봇 홈 포즈.**
      - 중력 정렬: ARCore +Y(위) -> 로봇 베이스 +Z(위). 이건 임의값이 아니다
      - 방위(yaw): 첫 프레임 광축의 수평 성분 -> 로봇 +X
      - 평행이동: 첫 프레임 파지점 -> 홈 파지점 좌표

    **대가**: 물체의 절대 위치 다양성이 **카메라 시점 다양성으로만** 표현된다.
    그리고 실물 실행 때 로봇이 **항상 같은 홈 포즈에서 출발**해야 한다.
    손목 카메라가 물체 상대위치를 이미지에 담으므로 델타 정책은 이 조건에서 학습된다.
    보드가 생기면 **재변환한다** — 그래서 `notes` 에 우회 사실을 박아 둔다.

축 규약 (전부 번들이 스스로 적은 것이거나 HW 회신이다 🔵):
    poses.csv  : ARCore/OpenGL. x 오른쪽 · y 위 · -z 앞. 쿼터니언 xyzw, camera-local -> world
    T_cam_pinch: OpenCV. x 오른쪽 · y 아래 · z 앞. **저장된 JPEG 기준** (뒤집힌 실장착)
    RawEpisode : 쿼터니언 **wxyz**
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from umi.raw import BUNDLE_SCHEMA_SUPPORTED, RawEpisode, RawError, RawMeta

__all__ = ["PilotCalib", "to_raw_pilot", "HANDEYE_MEASURED"]

# ARCore(GL) -> OpenCV 카메라. y, z 부호 반전. HW 회신의 M = diag(1,-1,-1) 과 같다 🔵
GL_TO_CV = np.diag([1.0, -1.0, -1.0])

# --- 스키마 경계 (D-AI-41) ----------------------------------------------------
# 앱이 실제로 내는 이름. 번들 71편 전부 이 값이다 🟢
# (`MEASURE_umi_bundle_intake_0910.md` 「부수 차이」).
ARPOSE_SCHEMA_ACCEPTED = ("arpose.episode/1",)
# 이 어댑터가 내보내는 이름. `raw.py` 가 받는 유일한 값이다.
CANONICAL_BUNDLE_SCHEMA = "umi_raw/0.1.0"
assert CANONICAL_BUNDLE_SCHEMA in BUNDLE_SCHEMA_SUPPORTED

# 실측·검정된 T_cam->pinch. MEASURE_handeye_from_video_0912.md 🟢
# 회전은 HW 도면 + 뒤집힌 장착, 평행이동은 영상 5,149프레임.
HANDEYE_MEASURED = np.array([
    [1.0,  0.0,     0.0,    -0.0014],
    [0.0, -0.259,   0.966,  -0.0401],
    [0.0, -0.966,  -0.259,   0.1497],
    [0.0,  0.0,     0.0,     1.0],
])


@dataclass
class PilotCalib:
    """Everything the pilot needs that is not in the bundle.
    번들에 없고 파일럿이 필요로 하는 전부. **기본값을 주지 않는다** --
    근사값이 조용히 정본이 되는 것을 막는다."""

    t_cam_pinch: np.ndarray          # (4,4) 저장 JPEG OpenCV 기준
    anchor_pos: np.ndarray           # (3,) 로봇 베이스 기준 앵커 좌표 [m]
    calibration_id: str              # 빈 값이면 validate_raw 가 실기록을 거부한다
    anchor_at: str = "grasp"         # "grasp" | "first"


def _quat_xyzw_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        raise RawError("쿼터니언 노름이 0 이다")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Shepperd's method -- picks the largest denominator so no branch loses precision."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _base_from_anchor(
    T_world_pinch: np.ndarray,
    anchor_pos: np.ndarray,
    grasp_index: int | None = None,
    anchor_at: str = "grasp",
) -> np.ndarray:
    """T_base<-world, defined so frame 0's pinch sits at the robot's home pinch pose.
    첫 프레임 파지점이 로봇 홈 파지점에 오도록 정의한 T_base<-world.

    세 성분으로 쪼갠다. **중력만이 물리적으로 결정되고 나머지 둘은 정의다.**
      1) 중력 정렬  ARCore +Y(위) -> 베이스 +Z(위).  물리 🟢
      2) 방위(yaw)  홈 -> **폐쇄 시점 파지점** 의 수평 방향 -> 베이스 +X.  정의 🟡
      3) 평행이동   **앵커 프레임**의 파지점 -> anchor_pos.  정의 🟡

    앵커 이력 (2026-09-12):
      1차는 첫 프레임을 로봇 홈에 박았다. 그러면 폐쇄 지점이 홈에서 시연자가 이동한
      거리만큼 떨어지는데, 실측상 그게 0.30m 를 넘어 **로봇 도달 반경 밖**이 됐다 🟢.
      2차는 **폐쇄 시점**을 앵커로 쓴다. 정책이 실제로 정밀해야 하는 곳이 거기이므로
      그 지점을 작업영역 한가운데 박고, 접근 구간이 어디서 오든 그건 부차적이다.

    yaw 정의 이력 (2026-09-12):
      1차는 "첫 프레임 접근축의 수평 성분"이었다. **틀렸다.** 접근축은 중력과 1.2도로
      거의 수직이라 수평 성분이 0 에 가깝고, 코드가 턱축 fallback 을 탔다. 턱축은 손목
      회전에 따라 임의 방향이라 궤적이 아무 데나 놓였다 -- 1편 스모크에서 프레임 68%가
      도달 영역 밖으로 나왔다 🟢.
      2차는 **물체가 있는 방향**을 쓴다. 폐쇄 시점의 파지점이 곧 물체 위치이므로,
      홈에서 그쪽을 향하는 수평 방향을 +X 로 놓으면 물체가 항상 로봇 정면에 온다.
      폐쇄가 없는 편은 첫->끝 수평 변위로 대신한다.
    """
    # 1) 중력 정렬: world(x, y, z) -> base(x, -z, y). Y_world -> Z_base 가 되는 최소 회전.
    R_grav = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [0.0, 1.0, 0.0]])

    p0 = R_grav @ T_world_pinch[0][:3, 3]
    target_i = grasp_index if grasp_index is not None else len(T_world_pinch) - 1
    target_i = max(1, min(int(target_i), len(T_world_pinch) - 1))
    p_t = R_grav @ T_world_pinch[target_i][:3, 3]
    anchor_i = target_i if (anchor_at == "grasp" and grasp_index is not None) else 0

    horiz = np.array([p_t[0] - p0[0], p_t[1] - p0[1], 0.0])
    if np.linalg.norm(horiz) < 1e-3:      # 1mm 미만이면 방향이라 할 수 없다
        # 제자리 파지다. 궤적 전체의 수평 변위로 대신한다.
        p_end = R_grav @ T_world_pinch[-1][:3, 3]
        horiz = np.array([p_end[0] - p0[0], p_end[1] - p0[1], 0.0])
        if np.linalg.norm(horiz) < 1e-3:
            raise RawError("yaw 를 정할 수평 변위가 없다 (전 구간 1mm 미만)")
    horiz /= np.linalg.norm(horiz)
    # 2) 그 수평 방향이 베이스 +X 가 되도록 z 축 둘레로 돌린다.
    c, s = float(horiz[0]), float(horiz[1])
    R_yaw = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    R_bw = R_yaw @ R_grav
    t_bw = (np.asarray(anchor_pos, dtype=np.float64)
            - R_bw @ T_world_pinch[anchor_i][:3, 3])
    T = np.eye(4)
    T[:3, :3] = R_bw
    T[:3, 3] = t_bw
    return T


def _read_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _read_gripper(path: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    gap = np.full(n, np.nan, dtype=np.float64)
    status = np.full(n, "X", dtype="<U1")
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            i = int(row["frame_index"])
            if not (0 <= i < n):
                continue
            st = (row.get("status") or "X").strip()[:1] or "X"
            status[i] = st
            v = (row.get("gap_m") or "").strip()
            # D·M 만 값이 있다. T·X 는 채우지 않는다 -- 0 으로 채우면 학습이
            # "완전히 닫혔다" 를 배운다 (SPEC · track_a/convert/arcore.py).
            if v and st in ("D", "M"):
                gap[i] = float(v)
    return gap, status


def to_raw_pilot(
    bundle: Path,
    calib: PilotCalib,
    *,
    gripper_csv: str = "gripper.csv",
    image_size: int | None = 224,
    skill_id: str | None = None,
    trim_after_grasp: int | None = 15,
    pre_grasp_frames: int | None = 45,
) -> RawEpisode:
    """One ARCore bundle directory -> RawEpisode in robot_base frame. PILOT.

    `image_size` 를 주면 그 정사각형으로 줄인다. RawEpisode 규약은 원본 해상도
    보존이지만 71편 x 1920x1080 은 33GB 라 파일럿에서는 줄인다 -- **이 사실을
    `meta.notes` 에 박는다.** None 이면 원본 그대로.

    `pre_grasp_frames` 를 주면 폐쇄 **이전** 그만큼만 남긴다 (기본 45프레임 = 1.5초).
    근거: 그 앞 구간은 사람이 물체 쪽으로 걸어가는 긴 수평 이동이고 **화면에 물체가 없다.**
    학습 대상이 아니고, 실측상 그 구간이 로봇 도달 반경 0.3m 를 넘긴다 🟢
    (3편 스모크: 폐쇄까지 잘랐는데도 최대반경 0.315~0.354m).
    남는 구간은 "물체가 보이는 곳에서 내려가 닫는다" 이고, 그게 정책이 배울 것이다.

    `trim_after_grasp` 를 주면 **폐쇄 시점 + 그만큼** 에서 자른다 (기본 15프레임 = 0.5초).
    근거: D-AI-22 가 모든 스킬을 `[집기 = 학습 정책] + [놓기 = 스크립트]` 로 쪼갰다.
    정책이 배울 것은 집는 데까지이고, 그 뒤 사람이 물체를 들고 이동한 구간은 학습 대상이
    아니다. 그리고 실측상 그 구간이 **로봇 도달 반경 0.3m 를 넘긴다** 🟢 (3편 스모크에서
    최대반경 0.315~0.358m). 자르면 두 문제가 같이 풀린다. None 이면 자르지 않는다.

    ⚠️ 이미지를 **회전하지 않는다.** 프레임은 180도 뒤집혀 저장돼 있고
    `T_cam->pinch` 도 그 뒤집힌 프레임 기준이다. 둘은 서로 일관되므로 손대지 않는다.
    실물 로봇 카메라를 같은 방향으로 달거나 추론 시점에 맞춰야 한다 (LIMITS).
    """
    from PIL import Image

    meta_j = json.loads((bundle / "episode.json").read_text(encoding="utf-8"))
    contract = meta_j.get("contract", {})
    if contract.get("quaternion_order") != "xyzw":
        raise RawError(f"쿼터니언 순서가 xyzw 가 아니다: {contract.get('quaternion_order')!r}")
    if contract.get("up_axis") != "Y":
        raise RawError(f"up_axis 가 Y 가 아니다: {contract.get('up_axis')!r}")

    rows = list(csv.DictReader((bundle / "poses.csv").open(encoding="utf-8")))
    if not rows:
        raise RawError(f"poses.csv 가 비었다: {bundle}")
    n_full = len(rows)

    # 그리퍼를 먼저 읽는다 -- 폐쇄 시점이 yaw 정의와 절단 지점 둘 다를 정한다.
    gap_full, status_full = _read_gripper(bundle / gripper_csv, n_full)
    grasp_i: int | None = None
    finite = gap_full[np.isfinite(gap_full)]
    if finite.size >= 4:
        mid = (float(finite.max()) + float(finite.min())) / 2.0
        below = np.nonzero(np.isfinite(gap_full) & (gap_full < mid))[0]
        if below.size:
            grasp_i = int(below[0])

    trimmed_from = n_full
    lo, hi = 0, n_full
    if grasp_i is not None:
        if trim_after_grasp is not None:
            hi = min(n_full, grasp_i + int(trim_after_grasp) + 1)
        if pre_grasp_frames is not None:
            lo = max(0, grasp_i - int(pre_grasp_frames))
        if hi - lo < 8:                     # 너무 짧으면 에피소드가 못 쓰게 된다
            lo, hi = 0, n_full
    if (lo, hi) != (0, n_full):
        rows = rows[lo:hi]
        gap_full = gap_full[lo:hi]
        status_full = status_full[lo:hi]
        grasp_i = grasp_i - lo if grasp_i is not None else None

    n = len(rows)
    gap, status = gap_full, status_full

    pos = np.zeros((n, 3), dtype=np.float64)
    quat = np.zeros((n, 4), dtype=np.float64)
    stamps = np.zeros(n, dtype=np.float64)
    T_world_pinch = np.zeros((n, 4, 4), dtype=np.float64)

    for i, r in enumerate(rows):
        R_gl = _quat_xyzw_to_R(float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"]))
        T_wc = np.eye(4)
        T_wc[:3, :3] = R_gl @ GL_TO_CV      # world <- camera(OpenCV)
        T_wc[:3, 3] = [float(r["x"]), float(r["y"]), float(r["z"])]
        T_world_pinch[i] = T_wc @ calib.t_cam_pinch
        stamps[i] = float(r["timestamp_ns"]) * 1e-9

    T_bw = _base_from_anchor(T_world_pinch, calib.anchor_pos, grasp_i, calib.anchor_at)
    for i in range(n):
        T = T_bw @ T_world_pinch[i]
        pos[i] = T[:3, 3]
        quat[i] = _R_to_quat_wxyz(T[:3, :3])

    imgs: list[np.ndarray] = []
    img_stamps: list[float] = []
    for i, r in enumerate(rows):
        name = (r.get("image") or "").strip()
        if not name:
            continue
        p = bundle / "frames" / name
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        if image_size:
            im = im.resize((image_size, image_size))
        imgs.append(np.asarray(im, dtype=np.uint8))
        img_stamps.append(stamps[i])

    if not imgs:
        raise RawError(f"이미지가 하나도 없다: {bundle}")

    s = _read_summary(bundle / "summary.txt")
    rate = float(s.get("rate_hz", 30.0))
    n_missing = n - len(imgs)

    bundle_skill = str(meta_j.get("session", {}).get("skill_id", ""))
    notes: dict[str, Any] = {
        "pilot": True,
        # 번들이 뭐라고 적었는지 그대로 남긴다. 계약 enum 으로 옮긴 것이라면
        # **무엇을 무엇으로 옮겼는지**가 데이터에 붙어 있어야 나중에 되짚어진다.
        "bundle_skill_id": bundle_skill,
        "base_frame_definition": f"anchor_at={calib.anchor_at} pinch = {list(calib.anchor_pos)}",
        "issue_29_unresolved": (
            "T_ARCore월드->로봇베이스 미결. 중력 정렬만 물리이고 yaw·평행이동은 정의다. "
            "작업대 기준(ArUco 보드 등)이 생기면 재변환해야 한다"
        ),
        "handeye_source": "MEASURE_handeye_from_video_0912.md",
        "image_rotated": False,
        "image_note": "프레임이 180도 뒤집혀 저장됐고 T_cam->pinch 도 같은 기준이라 회전하지 않았다",
        "image_resized_to": image_size,
        "pre_stabilized": s.get("pre_stabilized", "unknown"),
        "gripper_csv": gripper_csv,
        "yaw_definition": "home -> grasp-moment pinch, horizontal",
        "grasp_index": grasp_i,
        "trimmed_after_grasp": trim_after_grasp,
        "pre_grasp_frames": pre_grasp_frames,
        "trim_window": [lo, hi],
        "n_steps_before_trim": trimmed_from,
        "trim_reason": (
            "D-AI-22 가 스킬을 [집기=정책]+[놓기=스크립트] 로 쪼갰다. 폐쇄 이후는 놓기이고, "
            "폐쇄 한참 이전은 물체가 화면에 없는 이동 구간이다. 둘 다 학습 대상이 아니고 "
            "실측상 로봇 도달 반경 0.3m 를 넘긴다"
        ),
        "bundle_intrinsics": meta_j.get("camera", {}).get("intrinsics", {}),
    }
    if image_size:
        notes["image_resize_deviation"] = (
            "RawEpisode 규약은 원본 해상도 보존이다. 파일럿 용량(71편 x 1920x1080 = 약 33GB) "
            "때문에 줄였다 -- 이 값으로 학습한 수치를 원본 해상도 결과와 섞지 않는다"
        )

    seg: list[list[int]] = []
    if s.get("usable_segments"):
        try:
            lo, hi = s["usable_segments"].split("~")
            seg = [[int(float(lo) * rate), int(float(hi) * rate)]]
        except Exception:  # noqa: BLE001 - 형식이 다르면 비워 둔다
            seg = []

    mapped_skill = skill_id or bundle_skill
    if bundle_skill and mapped_skill != bundle_skill:
        notes["skill_id_mapped_from"] = bundle_skill

    # --- 경계 어댑터: 앱 원본 스키마 -> canonical -----------------------------
    # 이 함수가 그 경계다. `raw.py` 는 canonical(`umi_raw/0.1.0`) 만 받는다.
    # 여기서 이름을 바꾸되 원본을 notes 에 남겨 출처를 잃지 않는다.
    # 트랙 A `track_a/convert/arpose_delivery.py` 가 병합되면 그쪽이 정본이 되고
    # 이 블록은 지운다 (D-AI-41).
    origin_schema = str(meta_j.get("schema", ""))
    if origin_schema and origin_schema not in BUNDLE_SCHEMA_SUPPORTED:
        if origin_schema not in ARPOSE_SCHEMA_ACCEPTED:
            raise ValueError(
                f"모르는 번들 스키마 {origin_schema!r} 다. "
                f"이 어댑터가 canonical 로 바꿀 수 있는 것은 {ARPOSE_SCHEMA_ACCEPTED} 뿐이다 — "
                "앱이 형식을 바꿨다면 어댑터를 먼저 고쳐라"
            )
        notes["bundle_schema_origin"] = origin_schema

    rm = RawMeta(
        recording_id=meta_j.get("episode", bundle.name),
        skill_id=mapped_skill,
        source="arcore",
        frame="robot_base",
        n_steps=n,
        pose_rate_hz=rate,
        cameras=["cam_wrist"],
        calibration_id=calib.calibration_id,
        bundle_schema=CANONICAL_BUNDLE_SCHEMA,
        frames_dropped=int(s.get("frames_dropped", n_missing)),
        usable_segments=seg,
        device=str(meta_j.get("device", "")),
        recorded_by="UMI handheld (pilot)",
        notes=notes,
    )
    return RawEpisode(
        meta=rm,
        eef_pos=pos,
        eef_quat=quat,
        gripper_gap_m=gap,
        gripper_status=status,
        pose_timestamp=stamps,
        images={"cam_wrist": np.stack(imgs)},
        image_timestamp={"cam_wrist": np.asarray(img_stamps, dtype=np.float64)},
    )
