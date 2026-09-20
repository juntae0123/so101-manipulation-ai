"""Closed-loop policy runner: camera -> policy -> IK -> SO-101 joints.
폐루프 러너. 카메라 → 정책 → IK → SO-101 관절.

제품 기조 (이걸 틀리면 나머지가 다 틀어진다)
--------------------------------------------
**결과 모방이지 궤적 모방이 아니다.** 시연 궤적을 그대로 그리는 게 목표가 아니라
같은 물체를 같게 집는 게 목표다. 접근 경로는 IK 가 스스로 푼다.

그래서 이 도구는 **재관측한다.** 8스텝 중 [1,5) 만 보내고 다시 본다(0.4초마다).
전체 궤적을 미리 풀어 한 번에 재생하면 오차가 누적되고, 무엇보다 **궤적 모방을
재는 셈이 된다.**

⚠️ 판정은 궤적 오차가 아니라 **파지 성공**이다 — 물체가 10cm 이상 올라가고 0.5초 유지
(D-AI-58 의 1b 게이트).

카메라 소스 — 하나만 고른다
---------------------------
    --camera v4l2:0                 USB 캠 / 폰을 웹캠으로 붙인 경우
    --camera dir:/path/frames       폴더의 프레임을 순서대로 (폰이 올려주는 방식)
    --camera zarr:/path.zarr.zip:3  기록된 시연 프레임 (**로봇 없이 배선 전체 검증용**)

`zarr:` 로 `--no-robot --dry-run` 을 돌리면 카메라도 팔도 없이 전 구간이 도는지 본다.
월요일에 카메라만 꽂으면 되게 하려면 이걸 먼저 통과시킨다.

전처리 — evaluate.py 와 같은 경로를 쓴다
----------------------------------------
    tf = get_image_transform(in_res=(W, H), out_res=(224, 224))
    rgb = draw_predefined_mask(rgb, color=(0,0,0), mirror=False, gripper=True, finger=False)
    obs = get_real_umi_obs_dict(raw, shape_meta, obs_pose_repr=cfg..., episode_start_pose=[start])
    absolute = get_real_umi_action(action, raw, action_pose_repr=cfg...)   # ← 'relative'

⚠️ `action_pose_repr` 를 안 넘기면 기본값 `'abs'` 로 떨어져 팔이 원점으로 간다.
   `'rel'` 은 소스가 스스로 legacy buggy 라 적은 별개 경로다. **셋 다 에러 없이 돈다.**

Usage
-----
  python run_policy_realtime.py --selftest
  # 배선 검증 (로봇·카메라 없이)
  python run_policy_realtime.py --checkpoint CK --task configs/can_side.yaml \
      --camera zarr:outputs/ds_f0918_42_holdout.zarr.zip:0 --no-robot --dry-run --steps 12
  # 실물
  python run_policy_realtime.py --checkpoint CK --task configs/can_side.yaml \
      --camera v4l2:0 --port /dev/ttyTHS1 --yes
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ACTION_STEPS = 4                 # evaluate.py 기본값. index [1, 1+4)
MAX_STEP_M = 0.035               # evaluate.py 의 컨트롤러 상한. 한 번에 이만큼만 움직인다
Z_CLIP = (0.006, 0.40)


# ──────────────────────────────────────────────── 카메라 소스

class DirSource:
    """Frames from a directory, in sorted order. 폴더의 프레임을 정렬 순서로."""

    def __init__(self, path: str) -> None:
        self.files = sorted(Path(path).glob("*.[jp][pn]g"))
        if not self.files:
            raise SystemExit(f"!! {path} 에 jpg/png 가 없다 (찾음 0 / 전체 "
                             f"{len(list(Path(path).glob('*')))})")
        self.i = 0
        print(f"[카메라] dir {path} · 프레임 {len(self.files)}장")

    def read(self):
        import cv2
        if self.i >= len(self.files):
            return None
        img = cv2.imread(str(self.files[self.i]))
        self.i += 1
        return None if img is None else img[:, :, ::-1].copy()


class ZarrSource:
    """Recorded demo frames. 기록된 시연 프레임. 로봇 없이 배선을 검증한다."""

    def __init__(self, spec: str) -> None:
        path, _, ep = spec.rpartition(":")
        import zarr
        z = zarr.open(path, mode="r")
        ends = np.asarray(z["meta"]["episode_ends"][:])
        e = int(ep)
        if not (0 <= e < len(ends)):
            raise SystemExit(f"!! 에피소드 {e} 가 범위 밖 (편 {len(ends)}개)")
        self.lo = 0 if e == 0 else int(ends[e - 1])
        self.hi = int(ends[e])
        self.arr = z["data"]["camera0_rgb"]
        self.pos = z["data"]["robot0_eef_pos"]
        self.rot = z["data"]["robot0_eef_rot_axis_angle"]
        self.gw = z["data"]["robot0_gripper_width"]
        self.i = self.lo
        self.last = self.lo
        print(f"[카메라] zarr {path} 편 {e}/{len(ends)} · 행 {self.lo}~{self.hi} "
              f"({self.hi - self.lo}프레임)")

    def state(self, row: int) -> dict:
        """Recorded proprioception for one row. 그 행의 기록된 자기수용 상태."""
        return {"robot0_eef_pos": np.asarray(self.pos[row], float),
                "robot0_eef_rot_axis_angle": np.asarray(self.rot[row], float),
                "robot0_gripper_width": np.asarray(self.gw[row], float).reshape(1)}

    def read(self):
        if self.i >= self.hi:
            return None
        self.last = self.i
        img = np.asarray(self.arr[self.i])
        self.i += 1
        if img.dtype != np.uint8:                      # 정규화돼 있으면 되돌린다
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[0] in (1, 3) and img.ndim == 3:   # CHW 로 저장된 경우
            img = np.transpose(img, (1, 2, 0))
        return img


class V4l2Source:
    def __init__(self, idx: int) -> None:
        import cv2
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            raise SystemExit(f"!! 카메라 {idx} 를 못 연다")
        print(f"[카메라] v4l2 {idx}")

    def read(self):
        ok, bgr = self.cap.read()
        return bgr[:, :, ::-1].copy() if ok else None


class HttpSource:
    """Snapshot-per-read from a phone running as an IP webcam.
    폰을 IP 웹캠으로 띄웠을 때, 읽을 때마다 스냅샷을 새로 받는다.

    Why snapshots and not the MJPEG stream / 왜 스트림이 아니라 스냅샷인가
    ------------------------------------------------------------------
    MJPEG 스트림을 열어두면 디코더 버퍼에 프레임이 쌓인다. 우리는 0.4초에 한 번만
    읽으므로, 읽을 때 나오는 것은 **몇 초 전 장면**일 수 있다. 그런데 낡은 프레임과
    새 프레임은 **똑같이 생긴 배열**로 나온다 — 검사가 안 되는 모양이다.
    스냅샷 GET 은 요청 시점 화면을 주므로 이 실패가 원천적으로 없다.

    그래도 스트림이 얼어붙는 경우가 있어서, 직전 바이트와 완전히 같은 응답이
    연속으로 오면 세어 두고 경고한다. **"얼었다"와 "정지한 장면"은 다르지만,
    구분이 안 되면 최소한 숫자로 보여야 한다.**

    IP 웹캠 앱은 보통 두 경로를 낸다 — `/shot.jpg`(스냅샷) 과 `/video`(MJPEG).
    **스냅샷 경로를 준다.** `/video` 를 주면 거부한다.
    """

    def __init__(self, url: str, timeout: float = 3.0) -> None:
        self.url = url
        self.timeout = timeout
        self.reads = 0
        self.repeats = 0
        self._last: bytes | None = None
        low = url.lower()
        if low.rstrip("/").endswith("/video") or "action=stream" in low:
            raise SystemExit(
                f"!! 스트림 경로다: {url}\n"
                "   MJPEG 스트림은 버퍼에 쌓인 낡은 프레임을 새 프레임처럼 준다.\n"
                "   스냅샷 경로를 줘라 (IP Webcam 앱이면 .../shot.jpg)")
        first = self._fetch()
        if first is None:
            raise SystemExit(f"!! 첫 스냅샷을 못 받았다: {url}")
        print(f"[카메라] http 스냅샷 {url} · 첫 프레임 {first.shape}")

    def _fetch(self):
        import urllib.request

        import cv2
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
                raw = r.read()
        except Exception as exc:                        # noqa: BLE001 — 사유를 남긴다
            print(f"   !! 스냅샷 실패 {type(exc).__name__}: {exc}")
            return None
        if not raw:
            print("   !! 스냅샷이 0바이트다")
            return None
        if raw == self._last:
            self.repeats += 1
        self._last = raw
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"   !! JPEG 디코드 실패 ({len(raw)} bytes)")
            return None
        return bgr[:, :, ::-1].copy()

    def read(self):
        self.reads += 1
        return self._fetch()

    def report(self) -> str:
        """Freshness with its denominator. 신선도를 모수와 함께."""
        return f"http 스냅샷 {self.reads}회 · 직전과 동일 바이트 {self.repeats}회"


def parse_camera(spec: str):
    """One of v4l2:<i> / dir:<path> / zarr:<path>:<ep> / http(s)://<snapshot-url>.
    넷 중 하나만."""
    usage = ("v4l2:0 / dir:/path / zarr:/path.zarr.zip:0 / "
             "http://192.168.0.5:8080/shot.jpg")
    kind, sep, rest = spec.partition(":")
    if kind in ("http", "https"):
        # URL 전체가 주소다. rest 만 떼면 스킴이 날아간다.
        if not rest.startswith("//") or len(rest) <= 2:
            raise SystemExit(f"!! URL 이 아니다: {spec!r}. 예) {usage.split(' / ')[-1]}")
        return HttpSource(spec)
    if not sep or not rest:
        raise SystemExit(f"!! --camera 형식이 아니다: {spec!r}. {usage}")
    if kind == "v4l2":
        return V4l2Source(int(rest))
    if kind == "dir":
        return DirSource(rest)
    if kind == "zarr":
        return ZarrSource(rest)
    raise SystemExit(f"!! 모르는 카메라 종류 {kind!r}. v4l2 / dir / zarr / http 중 하나")


# ──────────────────────────────────────────────── 관측 조립

def stack_history(hist: list[dict]) -> dict:
    """Stack a deque of per-step obs into (T, ...) arrays. 스텝별 관측을 (T, ...) 로 쌓는다."""
    if not hist:
        raise ValueError("관측 이력이 비었다")
    keys = set(hist[-1])
    for h in hist:
        if set(h) != keys:
            raise ValueError(f"이력마다 키가 다르다: {sorted(keys)} vs {sorted(h)}")
    return {k: np.stack([h[k] for h in hist]) for k in keys}


def clamp_target(target_pos: np.ndarray, current_pos: np.ndarray) -> np.ndarray:
    """Bound one commanded step, exactly as evaluate.py does. evaluate.py 와 같은 상한."""
    delta = np.asarray(target_pos, float) - np.asarray(current_pos, float)
    n = float(np.linalg.norm(delta))
    if n > MAX_STEP_M:
        delta = delta * (MAX_STEP_M / n)
    out = np.asarray(current_pos, float) + delta
    out[2] = float(np.clip(out[2], *Z_CLIP))
    return out


# ──────────────────────────────────────────────── 자체검증

def selftest() -> int:
    ok = total = 0

    def check(name, cond, detail=""):
        nonlocal ok, total
        total += 1
        ok += bool(cond)
        print(f"[{total}] {name:<46} {'OK' if cond else '!! 실패'}  {detail}")

    bad_specs = ("v4l2", "", "webcam:0", "zarr:onlypath",
                 "http:", "http://", "https://192.168.0.5:8080/video")
    rejected = 0
    for bad in bad_specs:
        try:
            parse_camera(bad)
        except BaseException:                          # noqa: BLE001 — 죽기만 하면 된다
            rejected += 1
    check(f"잘못된 --camera 거부 {rejected} / {len(bad_specs)}",
          rejected == len(bad_specs))

    # 판별력 행: 형식이 맞는 URL 은 형식 검사를 통과해야 한다. 전부 거부하면
    # 위 행은 공짜로 통과한다 — 그건 검사가 아니다. 네트워크는 안 탄다.
    kind, sep, rest = "http://192.168.0.5:8080/shot.jpg".partition(":")
    check("판별력: 정상 스냅샷 URL 은 형식 검사를 통과한다",
          kind == "http" and rest.startswith("//") and len(rest) > 2)

    h = [{"a": np.zeros(3), "b": np.ones(1)} for _ in range(2)]
    s = stack_history(h)
    check("이력 2개 → (2, ...) 로 쌓임", s["a"].shape == (2, 3) and s["b"].shape == (2, 1),
          str({k: v.shape for k, v in s.items()}))

    try:
        stack_history([{"a": np.zeros(3)}, {"b": np.zeros(3)}])
        raised = False
    except ValueError:
        raised = True
    check("이력 키가 다르면 거부", raised)

    try:
        stack_history([])
        raised = False
    except ValueError:
        raised = True
    check("빈 이력 거부", raised)

    cur = np.array([0.40, 0.0, 0.05])
    far = cur + np.array([0.2, 0.0, 0.0])
    out = clamp_target(far, cur)
    check("한 스텝 상한 35mm", abs(np.linalg.norm(out - cur) - MAX_STEP_M) < 1e-12,
          f"{np.linalg.norm(out - cur) * 1000:.1f} mm")

    near = cur + np.array([0.01, 0.0, 0.0])
    out = clamp_target(near, cur)
    check("상한 안이면 그대로 (판별력)", np.allclose(out, near),
          f"{np.linalg.norm(out - cur) * 1000:.1f} mm")

    # ⚠️ 순서가 있다 — 35mm 상한이 **먼저**, z 클립이 나중이다 (evaluate.py 와 동일).
    #    초판은 z=-0.5 로 검사했는데 상한이 먼저 걸려 0.015 에서 멈춘다. 클립이
    #    물리지 않는 입력으로 클립을 검사한 셈이었다. 상한 안쪽 입력으로 바꾼다.
    low = clamp_target(np.array([0.40, 0.0, 0.0]), np.array([0.40, 0.0, 0.02]))
    check("z 하한으로 클립 (상한 안쪽 입력)", abs(low[2] - Z_CLIP[0]) < 1e-12,
          f"{low[2]:.4f} m  기대 {Z_CLIP[0]}")

    deep = clamp_target(np.array([0.40, 0.0, -0.5]), cur)
    check("상한이 z 클립보다 먼저 (판별력)", abs(deep[2] - (cur[2] - MAX_STEP_M)) < 1e-12,
          f"{deep[2]:.4f} m — 35mm 만 내려간다")

    # 액션 규약은 policy_to_joints 와 **같은 구현을 쓴다**. 두 벌이면 갈린다
    try:
        import policy_to_joints as p2j
        ident = np.zeros((8, 10))
        ident[:, 3], ident[:, 7], ident[:, 9] = 1.0, 1.0, 0.05
        T = np.eye(4); T[:3, 3] = [0.4, 0, 0.05]
        poses, gaps, diag = p2j.decode_chunk(ident, T)
        check("policy_to_joints 규약 공유 (항등 → 불변)",
              max(np.abs(q - T).max() for q in poses) < 1e-12 and diag["emitted"] == 4)
        check("실행 구간이 [1,5) 로 같다", diag["exec_slice"] == [1, 5])
    except ImportError as e:
        check(f"policy_to_joints import ({e})", False)

    try:
        from smoke_deploy_ckpt import _prepare_umi_path            # noqa: F401
        check("UMI 경로 해결기를 재사용한다 (두 벌 금지)", True)
    except ImportError as e:
        check(f"smoke_deploy_ckpt._prepare_umi_path import ({e})", False)

    check("프레임 stride 기본값 = 실행 스텝 수 (시간축 일치)", ACTION_STEPS == 4,
          f"{ACTION_STEPS}")
    # [14-16] bus.move() 경계 — mock publisher 로 **이 파일의** 전송 경계를 검사한다.
    #         so101_infer 의 판별행은 그쪽 경계만 본다 (황도경 지적 2026-09-20).
    class _Bus:
        def __init__(self): self.calls = []
        def move(self, i, t): self.calls.append((i, t))

    class _RT:
        JOINTS = [("j1", 1), ("j2", 2), ("j3", 3), ("j4", 4), ("j5", 5)]
        GRIPPER = ("grip", 6)
        @staticmethod
        def preflight(qs, gaps, dt): return {"ok": True}
        @staticmethod
        def joint_to_tick(j, r): return int(2048 + r * 100)
        @staticmethod
        def gripper_to_tick(g): return int(1763 - g * 10000)

    _plan = [(np.zeros(5), 0.05, 0.1, np.zeros(3)) for _ in range(4)]
    b1 = _Bus(); n1, _ = emit_chunk(_plan, "프리플라이트 불합격", b1, _RT(), 0.0, False)
    check("14 청크 거부 -> 전송 0회", n1 == 0 and len(b1.calls) == 0,
        f"bus.move {len(b1.calls)}회")
    b2 = _Bus(); n2, _ = emit_chunk(_plan, None, b2, _RT(), 0.0, False)
    check("15 청크 통과 -> 전송 판별행", n2 == 4 and len(b2.calls) == 4 * 6,
        f"{n2}점 · bus.move {len(b2.calls)}회 (기대 24)")
    b3 = _Bus(); n3, _ = emit_chunk(_plan, None, b3, _RT(), 0.0, True)
    check("16 dry-run -> 전송 0회", n3 == 0 and len(b3.calls) == 0, f"bus.move {len(b3.calls)}회")


    print(f"\n자체검증 {ok} / {total}")
    return 0 if ok == total else 1


# ──────────────────────────────────────────────── 본체

def emit_chunk(plan, chunk_reject, bus, rt, dt: float, dry_run: bool):
    """Send a chunk only if the whole chunk passed. 청크 전체가 통과해야만 보낸다.

    반환 (전송 횟수, 중단 사유). all-or-nothing 이다 — 초판은 웨이포인트별 즉시 전송이라
    청크 후반이 실패해도 앞 점들이 이미 나갔다 (황도경 지적 2026-09-20).
    이 함수가 run_policy_realtime -> bus.move() 경계 그 자체이고, 자체검사가 mock bus 로
    **여기를** 검증한다. so101_infer 의 판별행은 그쪽 경계만 본다.
    """
    if chunk_reject is not None:
        return 0, chunk_reject                      # 거부된 청크는 한 점도 안 나간다
    if bus is None or dry_run:
        return 0, None
    sent = 0
    for item in plan:
        q_new, gap_new = item[0], item[1]
        rrep = rt.preflight([list(q_new)], [gap_new], dt)
        ok = True if rrep is None else (rrep.get("ok", False)
                                        if isinstance(rrep, dict) else bool(rrep))
        if not ok:
            detail = rrep.get("fail") if isinstance(rrep, dict) else rrep
            return sent, f"replay_trajectory preflight 불합격 {detail}"
        for j, r in zip(rt.JOINTS, q_new):
            bus.move(j[1], rt.joint_to_tick(j, r))
        bus.move(rt.GRIPPER[1], rt.gripper_to_tick(gap_new))
        sent += 1
        time.sleep(dt)
    return sent, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--task")
    ap.add_argument("--camera",
                    help="v4l2:0 / dir:/path / zarr:/path.zarr.zip:0 / "
                         "http://192.168.0.5:8080/shot.jpg (폰 IP 웹캠 스냅샷 경로. "
                         "/video 같은 스트림 경로는 거부한다)")
    ap.add_argument("--port", default="/dev/ttyTHS1")
    ap.add_argument("--no-robot", action="store_true",
                    help="팔 없이 돈다. 관절 상태를 IK 결과로 이어붙여 배선만 검증")
    ap.add_argument("--dry-run", action="store_true", help="계산만. 서보에 아무것도 안 보낸다")
    ap.add_argument("--steps", type=int, default=40, help="재관측 횟수 상한")
    ap.add_argument("--action-steps", type=int, default=ACTION_STEPS)
    ap.add_argument("--jaw-tol-deg", type=float, default=10.0,
                    help="목표 자세와의 측지각 허용치[도]. policy_to_joints 기본값과 같다")
    ap.add_argument("--jaw-offset-deg", type=float, default=0.0)
    ap.add_argument("--frame-stride", type=int, default=None,
                    help="재관측 1회당 넘길 프레임 수. 기본 = --action-steps. "
                         "기록 소스(zarr/dir)의 시간축을 실행 주기와 맞춘다")
    ap.add_argument("--ik-reject-mm", type=float, default=None,
                    help="IK 위치 잔차가 이보다 크면 그 스텝을 버린다. "
                         "기본 없음 — evaluate.py 는 거부하지 않는다(97%%를 낸 그 경로). "
                         "실물에서 보수적으로 가고 싶을 때만 준다")
    ap.add_argument("--out", default=None)
    ap.add_argument("--umi-root", help="공식 UMI 저장소 경로 (diffusion_policy 의 부모)")
    ap.add_argument("--obs-from-zarr", action="store_true",
                    help="관측(자기수용)도 zarr 기록에서 준다. 교사강제. "
                         "팔 상태와 화면이 갈리지 않아 **정책이 닫는지**를 순수하게 본다. "
                         "제어를 재는 게 아니라 정책 출력을 재는 모드다")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    print("자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 팔을 건드리지 않는다")
    print()
    for need in ("checkpoint", "task", "camera"):
        if not getattr(a, need):
            ap.error(f"--{need} 가 필요하다")

    # ⚠️ 공식 UMI(`diffusion_policy`)를 sys.path 에 먼저 올린다. 안 하면 hydra 가
    #    'Error locating target' 로 죽는데 그건 체크포인트 문제가 아니라 환경 문제다.
    #    smoke_deploy_ckpt 의 해결기를 **재사용한다** — 두 벌이면 갈린다.
    from smoke_deploy_ckpt import _prepare_umi_path
    print(f"[{_prepare_umi_path(a.umi_root)}]")

    import torch, dill, hydra                                        # noqa: E401
    import policy_to_joints as p2j

    # 1) 정책
    payload = torch.load(a.checkpoint, map_location="cpu", pickle_module=dill,
                         weights_only=False)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    sd = payload["state_dicts"]
    key = "ema_model" if "ema_model" in sd else "model"
    policy.load_state_dict(sd[key])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(dev).eval()
    print(f"[정책] {key} · {dev} · horizon {cfg.shape_meta.action.horizon} · "
          f"실행 [1,{1 + a.action_steps})")

    obs_repr = cfg.task.pose_repr.obs_pose_repr
    act_repr = cfg.task.pose_repr.action_pose_repr
    print(f"[규약] obs_pose_repr={obs_repr} · action_pose_repr={act_repr}")
    if act_repr != "relative":
        raise SystemExit(f"!! action_pose_repr 가 {act_repr!r} 다. 'relative' 가 아니면 "
                         "상대를 절대로 해석한다. 중단")

    # 2) 로봇 모델 (IK·FK) + 파지점 게이트
    from simulation.env import PickEnv
    env = p2j.make_env(PickEnv, a.task)
    p2j.tcp_gate(p2j.probe_tcp_offset(env), None)

    from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action
    from umi.common.cv_util import draw_predefined_mask, get_image_transform

    cam = parse_camera(a.camera)
    first = cam.read()
    if first is None:
        raise SystemExit("!! 카메라에서 첫 프레임을 못 받았다")
    h, w = first.shape[:2]
    tf = get_image_transform(in_res=(w, h), out_res=(224, 224))
    print(f"[카메라] 입력 {w}x{h} → 224x224")

    # 3) 로봇
    # ⚠️ 2026-09-20 — 실물 전송 경로의 안전 임계값은 **여기서 정하지 않는다.**
    #    so101_infer 의 설정(AI/configs/real/so101_safety.json)에서만 읽고,
    #    누락·null·NaN 이면 버스를 열기 전에 죽는다 (fail-closed).
    #    초판은 이 경로에 상한이 아예 없어서, 현석 실물 로그의 wrist_roll -150.4도와
    #    arm_delta -1784(300틱 초과)를 우리 러너는 한 번도 검사하지 않았다.
    # ⚠️ 안전 임계값은 이 파일에서 정하지 않는다. so101_infer 의 설정에서만 읽는다.
    #    2026-09-20 2차 정정 — 초판은 이 로딩을 `not no_robot and not dry_run` 안에 두어
    #    dry-run 에서 상수가 None 인 채로 비교에 들어가 TypeError 가 났다 (황도경 지적).
    #    이제 **항상** 시도한다. 실패하면 실물 전송은 죽고, dry-run 은 "판정 불가"로 돈다 —
    #    "검사 안 함"이 "통과"로 보이지 않게 한다.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
    from so101_infer import load_safety, SafetyUnset, preflight as si_preflight
    SAFETY = None
    try:
        SAFETY = load_safety()
        print(f"안전 설정 로드: 임계 {len(SAFETY)}개 · "
              f"스텝 {int(SAFETY['max_step_tick'])}틱 · "
              f"속도 {SAFETY['max_joint_speed_rad_s']} rad/s · "
              f"여유 {SAFETY['min_joint_margin_deg']}도 · "
              f"wrist_roll [{math.degrees(SAFETY['wrist_roll_min_rad']):.1f}, "
              f"{math.degrees(SAFETY['wrist_roll_max_rad']):.1f}]도")
    except SafetyUnset as exc:
        if not a.no_robot and not a.dry_run:
            raise SystemExit(f"!! 안전 설정 미확정 — 실물 전송을 시작하지 않는다.\n{exc}")
        print(f"⚠️ 안전 설정 미확정 — 이번 실행의 안전 판정은 **판정 불가**다. 통과가 아니다.\n{exc}")

    bus = None
    if not a.no_robot:
        import replay_trajectory as rt
        bus = rt.Bus(a.port)
        if bus.position(200) is not None:
            bus.close(); raise SystemExit("!! 없는 ID 200 이 응답했다. 버스 이상. 중단")
        q_meas, g_meas, okn = rt.read_all(bus)
        if okn < len(rt.JOINTS):
            bus.close(); raise SystemExit(f"!! 관절 {okn}/{len(rt.JOINTS)} 만 읽힌다. 중단")
        q_cur = np.asarray(q_meas, float)
        # ⚠️ 2026-09-20 정정 — 초판은 g_meas 가 None 이면 0.09(완전 열림)로 **지어냈다.**
        #    위 건전성 검사는 팔 관절만 세므로 그리퍼 읽기 실패가 걸리지 않았고,
        #    그 값이 observe() 를 타고 정책 관측(robot0_gripper_width)으로 들어갔다.
        #    실제로는 물체를 쥐고 있는데 정책은 "아직 안 쥐었다"로 보고 접근을 다시 낸다.
        #    "못 읽었다"와 "열려 있다"가 같은 값이 되면 안 된다.
        if g_meas is None:
            bus.close()
            raise SystemExit("!! 그리퍼 폭을 못 읽었다. 이 값은 정책 관측으로 들어가므로 "
                             "추측값으로 대체하지 않는다. 서보 응답을 먼저 확인해라")
        gap_cur = float(g_meas)
        print(f"[로봇] 현재 자세 읽음 · gap {gap_cur * 1000:.1f} mm")
        if not (a.dry_run or a.yes):
            if input("팔 주변을 비웠나? 진행하려면 'go': ").strip() != "go":
                bus.close(); raise SystemExit("중단")
    else:
        q_cur = np.asarray(env.home_q, float)
        gap_cur = 0.09
        print("[로봇] --no-robot · 홈 자세에서 시작하고 IK 결과를 이어붙인다")

    import mujoco
    def fk(q):
        env.ik_data.qpos[env.qids] = q
        mujoco.mj_forward(env.model, env.ik_data)
        return env.tcp(env.ik_data).copy()

    from scipy.spatial.transform import Rotation
    T0 = fk(q_cur)
    episode_start = np.r_[T0[:3, 3], Rotation.from_matrix(T0[:3, :3]).as_rotvec()]

    teacher = a.obs_from_zarr
    if teacher and not isinstance(cam, ZarrSource):
        raise SystemExit("!! --obs-from-zarr 는 --camera zarr: 일 때만 된다")
    if teacher:
        print("[모드] 교사강제 — 자기수용도 기록에서 준다. 제어가 아니라 "
              "**정책이 닫는가**를 잰다")

    def observe(rgb):
        masked = draw_predefined_mask(rgb.copy(), color=(0, 0, 0), mirror=False,
                                      gripper=True, finger=False)
        if teacher:
            st = cam.state(cam.last)
            return {"camera0_rgb": tf(masked), **st}
        T = fk(q_cur)
        return {"camera0_rgb": tf(masked),
                "robot0_eef_pos": T[:3, 3].copy(),
                "robot0_eef_rot_axis_angle": Rotation.from_matrix(T[:3, :3]).as_rotvec(),
                "robot0_gripper_width": np.array([gap_cur])}

    hist = [observe(first)]
    hist.append(hist[0])
    log = []
    sent = 0
    rejected = 0
    gap_track = []
    try:
        for it in range(a.steps):
            raw = stack_history(hist[-2:])
            conv = get_real_umi_obs_dict(raw, cfg.shape_meta, obs_pose_repr=obs_repr,
                                         episode_start_pose=[episode_start])
            missing = set(cfg.shape_meta.obs) - set(conv)
            if missing:
                raise SystemExit(f"!! 관측 키 누락 {sorted(missing)} "
                                 f"({len(conv)} / {len(cfg.shape_meta.obs)})")
            batch = {k: torch.from_numpy(v.astype(np.float32))[None].to(dev)
                     for k, v in conv.items()}
            with torch.inference_mode():
                action = policy.predict_action(batch)["action"][0].cpu().numpy()
            absolute = get_real_umi_action(action, raw, action_pose_repr=act_repr)
            if teacher:
                gt_gap = float(cam.state(cam.last)["robot0_gripper_width"][0])
                pred_gap = float(np.clip(absolute[1][6], 0.0, 0.09))
                gap_track.append((cam.last, gt_gap * 1000, pred_gap * 1000))

            # ⚠️ 2026-09-20 2차 정정 (황도경 검토) — 초판 구조가 **웨이포인트별 즉시 전송**
            #    이라 청크 후반이 실패해도 앞 점들은 이미 나간 뒤였다. 그리고 안전 설정
            #    7개 중 max_step_tick·wrist_roll 만 실제로 검사하고 speed·joint_margin·
            #    min_tcp_z·ik_tol 은 읽기만 했다.
            #    → 4점을 **전부 먼저 풀고**, so101_infer.preflight 로 청크 통째로 검사한다.
            #      하나라도 불합격이면 그 청크의 전송은 0회다 (all-or-nothing).
            #      같은 함수를 쓰므로 설정 7개가 자동으로 전부 적용된다 — 오프라인 어댑터와
            #      받는 쪽이 **같은 자**를 쓴다.
            plan, chunk_reject = [], None
            q_walk = q_cur.copy()
            for target in absolute[1:1 + a.action_steps]:
                T_now = fk(q_walk)
                pos = clamp_target(target[:3], T_now[:3, 3])
                Rm = p2j.apply_jaw_offset(
                    Rotation.from_rotvec(target[3:6]).as_matrix(), a.jaw_offset_deg)
                try:
                    q_new = env.ik(pos, Rm, seed=q_walk, strict=False)
                except Exception as exc:                # noqa: BLE001
                    chunk_reject = f"IK 예외 {type(exc).__name__}"
                    break
                if q_new is None:
                    chunk_reject = "IK 해 없음"
                    break
                Tc = fk(q_new)
                res = float(np.linalg.norm(Tc[:3, 3] - pos))
                if a.ik_reject_mm is not None and res * 1000 > a.ik_reject_mm:
                    chunk_reject = f"IK 잔차 {res * 1000:.1f}mm > {a.ik_reject_mm:.0f}mm"
                    break
                gap_new = float(np.clip(target[6], 0.0, 0.09))
                # ⚠️ R3 (2026-09-20) — policy_to_joints.solve_waypoints 가 웨이포인트마다
                #    거는 측지각 검사가 실물 경로에는 없었다. 자세가 얼마나 틀어졌는지
                #    보지 않고 보내면, 파지 여유 ±25mm 인 계에서 확정적으로 빗나간다.
                e_rot = float(np.degrees(np.arccos(np.clip(
                    (np.trace(Tc[:3, :3].T @ Rm) - 1.0) / 2.0, -1.0, 1.0))))
                if e_rot > a.jaw_tol_deg:
                    chunk_reject = f"자세 측지각 {e_rot:.2f}도 > {a.jaw_tol_deg:.1f}도"
                    break
                plan.append((np.asarray(q_new, float), gap_new, res * 1000.0, pos))
                q_walk = np.asarray(q_new, float)

            if chunk_reject is None and not plan:
                chunk_reject = "경유점 0개"

            # 청크 전체 프리플라이트. 현재 자세를 맨 앞에 붙여야 첫 점의 스텝·속도도 잰다.
            rep = None
            if chunk_reject is None and SAFETY is not None:
                qs_chunk = [q_cur] + [pp[0] for pp in plan]
                gs_chunk = [gap_cur] + [pp[1] for pp in plan]
                rs_chunk = [0.0] + [pp[2] for pp in plan]
                rep = si_preflight(qs_chunk, gs_chunk, a.dt, rs_chunk, SAFETY)
                if not rep["ok"]:
                    chunk_reject = "프리플라이트 불합격 " + " · ".join(rep["fail"])
            elif chunk_reject is None and SAFETY is None:
                chunk_reject = ("안전 설정 미확정 — 검사할 자가 없다. "
                                "이 청크는 '통과'가 아니라 '판정 불가'다")

            _rt = None
            if bus is not None and not a.dry_run:
                import replay_trajectory as _rt
            ns, halt = emit_chunk(plan, chunk_reject, bus, _rt, a.dt, a.dry_run)
            sent += ns
            if chunk_reject is not None:
                print(f"  [{it}] 청크 거부 ({len(plan)}/{a.action_steps} 점 계산됨) — {chunk_reject}")
                print(f"        → 이 청크 전송 0회")
                rejected += max(1, len(plan))
            else:
                if halt:
                    print(f"  [{it}] {halt} — {ns}/{len(plan)} 점에서 중단")
                    rejected += 1
                for q_new, gap_new, res_mm, pos in plan[:ns or len(plan)]:
                    q_cur, gap_cur = q_new, gap_new
                    log.append({"iter": it, "pos": pos.tolist(), "gap_m": gap_new,
                                "q": [float(v) for v in q_new], "ik_residual_m": res_mm / 1000.0,
                                "preflight_ok": True})

            stride = a.frame_stride if a.frame_stride is not None else a.action_steps
            nxt = None
            for _ in range(stride):                 # 실행한 시간만큼 프레임을 넘긴다
                f = cam.read()
                if f is None:
                    break
                nxt = f
            if nxt is None:
                print(f"  프레임 끝 (반복 {it + 1}/{a.steps})")
                break
            hist.append(observe(nxt))
            hist = hist[-2:]
            if bus is not None and not a.dry_run:
                # ⚠️ 웨이포인트마다 a.dt 를 이미 기다린다. 여기서 또 0.4초를 자면
                #    재관측 주기가 0.8초가 된다 (황도경 지적 2026-09-20).
                #    실제로 실행한 시간만큼만 보정한다.
                spent = len(plan) * a.dt if not chunk_reject else 0.0
                remain = max(0.0, a.action_steps * a.dt - spent)
                if remain > 0:
                    time.sleep(remain)
    except KeyboardInterrupt:
        print("\n!! 중단됨")
    finally:
        if bus is not None:
            bus.close()

    print(f"\n반복 {len(set(r['iter'] for r in log))} · 명령 스텝 {len(log)} · "
          f"서보 전송 {sent}")
    if log:
        r = [x["ik_residual_m"] * 1000 for x in log]
        g = [x["gap_m"] * 1000 for x in log]
        print(f"IK 잔차  중앙 {np.median(r):.2f}mm  최대 {max(r):.2f}mm  "
              f"(거부 {rejected} / 시도 {len(log) + rejected})")
        print(f"gap      {min(g):.1f} ~ {max(g):.1f} mm")
        obj = 41.0
        if min(g) <= obj + 10:
            print(f"         → 물체 폭 {obj:.0f}mm 근처까지 닫힌다. 파지 동작이 있다")
        else:
            print(f"         → 최소 {min(g):.1f}mm 로 물체 폭 {obj:.0f}mm 보다 "
                  f"{min(g)-obj:.1f}mm 넓다. **이 구간에는 파지가 없다** "
                  f"(에피소드 끝까지 안 갔거나, 정책이 안 닫는 것이다)")
    else:
        print("!! 명령이 하나도 안 나갔다. IK 가 전부 거부됐거나 프레임이 없다")
    if not gap_track:
        print(f"\n=== 교사강제 gap 궤적 · 0 관측 ===")
        print("  --obs-from-zarr 를 안 줬다. 이 실행은 로봇 자세를 정책이 누적해서 만든다.")
        print("  이미지는 녹화 시연, 자세는 정책 — 두 입력이 다른 세계라 파지 판정에 쓸 수 없다.")
        print("  정책이 닫는지 보려면 --obs-from-zarr 로 다시 돌려라.")
    if gap_track:
        rows = [r for r, _, _ in gap_track]
        gt = np.array([g for _, g, _ in gap_track])
        pr = np.array([p for _, _, p in gap_track])
        print(f"\n=== 교사강제 gap 궤적 · {len(gap_track)} 관측 (행 {rows[0]}~{rows[-1]}) ===")
        print(f"  기록 {gt.min():.1f} ~ {gt.max():.1f} mm   (닫힘폭 {gt.max()-gt.min():.1f})")
        print(f"  예측 {pr.min():.1f} ~ {pr.max():.1f} mm   (닫힘폭 {pr.max()-pr.min():.1f})")
        print(f"  절대오차 중앙 {np.median(np.abs(pr-gt)):.2f} mm  최대 {np.abs(pr-gt).max():.2f} mm")
        obj = 41.0
        closes = pr.min() <= obj + 10
        print(f"  판정: 정책이 물체 폭 {obj:.0f}mm 근처까지 닫는가 → "
              f"{'닫는다' if closes else '**안 닫는다**'}")
        if gt.max() - gt.min() < 5:
            print("  ⚠️ 기록 자체의 닫힘폭이 5mm 미만이다. 이 편으로는 판정하지 마라")
        print("  " + "  ".join(f"{r}:{g:.0f}/{p:.0f}" for r, g, p in gap_track[:12]))
        print("  (행:기록/예측 mm)")
    if hasattr(cam, "report"):
        print("\n=== 카메라 ===")
        print("  " + cam.report())
        if getattr(cam, "repeats", 0):
            print("  ⚠️ 같은 바이트가 반복됐다. 장면이 정말 정지해 있었는지, "
                  "스트림이 얼었는지 구분되지 않는다. 화면을 흔들어 다시 확인해라")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"run": {"teacher_forcing": bool(teacher),
                     "camera": a.camera, "steps": a.steps,
                     "action_steps": a.action_steps,
                     "frame_stride": a.frame_stride,
                     "no_robot": bool(a.no_robot), "dry_run": bool(a.dry_run),
                     "jaw_offset_deg": a.jaw_offset_deg,
                     "ik_reject_mm": a.ik_reject_mm,
                     "checkpoint": a.checkpoint,
                     "n_log": len(log), "n_gap_track": len(gap_track),
                     "gap_track_empty_reason": (None if gap_track else
                         "teacher_forcing off — 기록 대상이 없다 (결측 아님)")},
             "log": log, "gap_track": gap_track}, indent=1), encoding="utf-8")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
