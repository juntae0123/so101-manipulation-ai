"""Adapter: UMI diffusion policy action chunk (relative EEF) -> SO-101 joint waypoints.
어댑터: UMI diffusion 정책의 상대 EEF 액션 청크를 SO-101 관절 웨이포인트로 바꾼다.

왜 필요한가 (2026-09-19)
------------------------
정책 출력은 `(8, 10)` 상대 EEF 궤적이고 로봇은 관절각을 받는다. 그 사이가 비어 있다.
`simulation/evaluate.py` 가 시뮬 안에서 이 변환을 이미 하고 있으므로 그 경로를 그대로
따르되, 실물 재생(`replay_trajectory.py`)이 먹는 형식으로 내놓는다.

    action (8,10)  ->  T_next = T_cur @ A_relative  ->  env.ik  ->  [[q1..q5], ...]
                                                                 +  gripper_m per waypoint

규약 — 전부 실측·소스 대조로 확정된 것만 쓴다 🟢
------------------------------------------------
    layout    [0:3] dx,dy,dz (m) · [3:9] rot6d · [9] gap (m, 절대)
    rot6d     회전행렬의 첫 두 **행** (열이 아니다)
              MEASURE_action_convention_0918.md §1 · 스윕 4후보 중 오차 2.22e-16 유일
    compose   T_next = T_cur @ A_relative
              diffusion_policy/common/pose_repr_util.py:62
    exec      index [1, 5) 만 실행하고 재관측 (evaluate.py:113 action_steps=4)
    gap       0 ~ 0.09 m. 0.15 초과는 mm 유입으로 보고 거부한다

⚠️ 이 도구가 **스스로 재고 거부하는 것** — 추정하지 않는다
-----------------------------------------------------------
1. **파지점 정의.** `env.tcp()` 는 MJCF `site tcp` 다. 그 site 가 gripper body 로컬
   어디인지는 어느 문서에도 없다. 저장소에는 후보가 둘이다.

       AI/configs/so101.yaml              pinch_offset_local [0,0,-0.080]        레거시
       AI/configs/grasp_so101_ver1.yaml   pinch_offset_local [0,0,-0.158118819]  ver1(실물)
       차이 78.118819 mm

   실물은 ver1 이다. 어댑터가 레거시 기준 모델로 IK 를 풀면 접근축으로 78.1mm 상수
   편향이 붙는다. 파지 여유는 ±25mm = (개구 90 − 물체 41)/2 뿐이고, hand-eye 계열
   상수 편향은 **학습이 지우지 못한다** (±10mm 에서 재생 3/4 🟢).

   → 이 도구는 site 위치를 **직접 재서** 두 후보와 대조한다. 어느 쪽도 아니면
     `--tcp-offset-m` 을 명시할 때까지 **궤적을 내놓지 않는다.**

2. **jaw 축.** ver1 jaw 는 레거시(+X) 기준 gripper 로컬 +Z 주위 +92.79도다
   (`grasp_so101_ver1.yaml`). 현행 IK 는 jaw 를 직접 구속하지 않는다.
   → `--jaw-offset-deg` 로 명시하지 않으면 **0 을 쓰고 그 사실을 리포트에 적는다.**
     92.79 를 조용히 넣지 않는다. 트랙 A 확인 대기 항목이다.

Usage
-----
  python policy_to_joints.py --selftest
  python policy_to_joints.py --task ~/handoff/configs/can_side.yaml --probe-tcp
  python policy_to_joints.py --task ~/handoff/configs/can_side.yaml \
      --action chunk.npy --from-home --out traj.json --out-replay traj_replay.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 규약 상수 — 바꾸려면 MEASURE 문서와 D- 기록이 먼저다
ACTION_DIM = 10
ROT6D_ROWS = True            # 첫 두 "행". 열이 아니다
GAP_MIN_M, GAP_MAX_M = 0.0, 0.09
GAP_REJECT_ABOVE_M = 0.15    # 단위 사고 감지: mm 가 들어오면 64.23 등으로 즉시 걸린다
EXEC_SLICE = (1, 5)          # evaluate.py 와 동일

# 파지점 후보 (gripper body 로컬 z, m)
TCP_CANDIDATES = {
    "legacy_so101.yaml": -0.080,
    "ver1_grasp_so101_ver1.yaml": -0.158118819,
}
TCP_MATCH_TOL_M = 0.002


# ---------------------------------------------------------------- 규약 변환

def rot6d_to_matrix(r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """Build a rotation matrix from the first two ROWS (UMI / v10 convention).
    첫 두 **행**에서 회전행렬을 만든다. 열로 읽으면 중앙오차 4mm 가 조용히 남는다."""
    a = np.asarray(r0, dtype=np.float64)
    b = np.asarray(r1, dtype=np.float64)
    n0 = np.linalg.norm(a)
    if n0 < 1e-9:
        raise ValueError("rot6d r0 의 노름이 0 이다")
    e0 = a / n0
    proj = b - np.dot(e0, b) * e0
    n1 = np.linalg.norm(proj)
    if n1 < 1e-9:
        raise ValueError("rot6d r0·r1 이 평행하다 — 회전 복원 불가")
    e1 = proj / n1
    e2 = np.cross(e0, e1)
    return np.stack([e0, e1, e2], axis=0)      # 행으로 쌓는다


def matrix_to_rot6d(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of rot6d_to_matrix (first two rows). 첫 두 행을 돌려준다."""
    R = np.asarray(R, dtype=np.float64)
    return R[0].copy(), R[1].copy()


def rot6d_residual(r0: np.ndarray, r1: np.ndarray) -> float:
    """How far the raw 6D pair is from an orthonormal pair, before Gram-Schmidt.
    직교정규화 **전** 원시 6D 쌍이 얼마나 어긋나 있는지. 입력 손상 감지용."""
    a = np.asarray(r0, float)
    b = np.asarray(r1, float)
    return float(max(abs(np.linalg.norm(a) - 1.0),
                     abs(np.linalg.norm(b) - 1.0),
                     abs(np.dot(a, b))))


def row_to_T(row: np.ndarray) -> np.ndarray:
    """One action row (10,) -> 4x4 relative transform. 액션 한 행을 상대 변환으로."""
    row = np.asarray(row, dtype=np.float64)
    if row.shape != (ACTION_DIM,):
        raise ValueError(f"행 차원 {row.shape} != ({ACTION_DIM},)")
    T = np.eye(4)
    T[:3, :3] = rot6d_to_matrix(row[3:6], row[6:9])
    T[:3, 3] = row[:3]
    return T


def compose(T_cur: np.ndarray, A_rel: np.ndarray) -> np.ndarray:
    """T_next = T_cur @ A_relative. 순서를 바꾸면 조용히 틀린다."""
    return np.asarray(T_cur, float) @ np.asarray(A_rel, float)


def decode_chunk(action: np.ndarray, T_cur: np.ndarray,
                 exec_slice: tuple[int, int] = EXEC_SLICE,
                 strict_gap: bool = True) -> tuple[list[np.ndarray], list[float], dict]:
    """Decode one (H,10) chunk into absolute TCP poses and gripper widths.
    청크 하나를 절대 TCP pose 와 개구 목록으로 푼다."""
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != ACTION_DIM:
        raise ValueError(f"액션 모양 {action.shape} — (H, {ACTION_DIM}) 이어야 한다")
    h = action.shape[0]
    lo, hi = exec_slice
    if not (0 <= lo < hi <= h):
        raise ValueError(f"exec_slice {exec_slice} 가 horizon {h} 범위 밖")

    worst_resid = 0.0
    poses: list[np.ndarray] = []
    gaps: list[float] = []
    for i in range(lo, hi):
        row = action[i]
        worst_resid = max(worst_resid, rot6d_residual(row[3:6], row[6:9]))
        poses.append(compose(T_cur, row_to_T(row)))
        g = float(row[9])
        if g > GAP_REJECT_ABOVE_M:
            raise ValueError(
                f"gap {g:.4f} > {GAP_REJECT_ABOVE_M} m — mm 단위가 흘러든 것으로 보고 거부한다")
        if strict_gap and not (GAP_MIN_M - 1e-6 <= g <= GAP_MAX_M + 1e-6):
            raise ValueError(f"gap {g:.4f} m 가 [{GAP_MIN_M}, {GAP_MAX_M}] 밖")
        gaps.append(float(np.clip(g, GAP_MIN_M, GAP_MAX_M)))

    diag = {
        "horizon": h,
        "exec_slice": list(exec_slice),
        "emitted": len(poses),
        "worst_rot6d_residual": worst_resid,
    }
    return poses, gaps, diag


def apply_jaw_offset(R: np.ndarray, deg: float) -> np.ndarray:
    """Rotate the target frame about its local Z (jaw opening direction).
    목표 프레임을 로컬 Z(턱 열림 방향) 주위로 돌린다. ver1 jaw 보정용."""
    if deg == 0.0:
        return R
    t = np.deg2rad(deg)
    Rz = np.array([[np.cos(t), -np.sin(t), 0.0],
                   [np.sin(t), np.cos(t), 0.0],
                   [0.0, 0.0, 1.0]])
    return np.asarray(R, float) @ Rz


# ---------------------------------------------------------------- 파지점 실측

def probe_tcp_offset(env) -> dict:
    """Measure where site `tcp` sits in the gripper body local frame, then classify.
    site `tcp` 가 gripper body 로컬 어디인지 재고 두 후보와 대조한다."""
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    d = env.data
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    if bid < 0:
        raise RuntimeError("body 'gripper' 를 찾지 못했다 — 모델이 다르다")
    Rb = d.xmat[bid].reshape(3, 3)
    pb = d.xpos[bid]
    local = Rb.T @ (d.site("tcp").xpos - pb)

    matches = {k: abs(float(local[2]) - v) for k, v in TCP_CANDIDATES.items()}
    best = min(matches, key=matches.get)
    ok = matches[best] <= TCP_MATCH_TOL_M
    return {
        "site_tcp_local_m": [float(v) for v in local],
        "z_local_m": float(local[2]),
        "candidates_m": TCP_CANDIDATES,
        "abs_diff_m": matches,
        "matched": best if ok else None,
        "tolerance_m": TCP_MATCH_TOL_M,
        "is_ver1": bool(ok and best.startswith("ver1")),
    }


def tcp_gate(probe: dict, override: float | None) -> None:
    """Refuse to emit a trajectory when the pinch point is not the real gripper's.
    파지점이 실물 것이 아니면 궤적을 내놓지 않는다."""
    if override is not None:
        print(f"[TCP] --tcp-offset-m {override} 명시됨 — 게이트 건너뜀 (사용자 책임)")
        return
    z = probe["z_local_m"]
    print(f"[TCP] site tcp 로컬 z = {z:+.9f} m")
    for k, v in TCP_CANDIDATES.items():
        print(f"      {k:<32} {v:+.9f}  차이 {abs(z - v) * 1000:8.3f} mm")
    if probe["matched"] is None:
        raise SystemExit(
            "!! 파지점이 두 후보 어느 쪽과도 "
            f"{TCP_MATCH_TOL_M * 1000:.0f}mm 안에서 맞지 않는다. 궤적을 내지 않는다.\n"
            "   --tcp-offset-m 으로 명시하거나 모델을 바꿔라.")
    if not probe["is_ver1"]:
        raise SystemExit(
            f"!! 모델의 파지점이 '{probe['matched']}' 다. 실물은 ver1 "
            f"({TCP_CANDIDATES['ver1_grasp_so101_ver1.yaml']:+.9f}) 이고 차이가 "
            f"{abs(TCP_CANDIDATES['ver1_grasp_so101_ver1.yaml'] - probe['z_local_m']) * 1000:.1f}mm 다.\n"
            "   이대로 실물에 올리면 접근축으로 그만큼 상수 편향이 붙는다.\n"
            "   파지 여유는 ±25mm 뿐이다. ver1 모델을 쓰거나 --tcp-offset-m 을 명시하라.")
    print("[TCP] ver1 파지점 확인. 통과")


# ---------------------------------------------------------------- IK

def solve_waypoints(env, poses, seed_q, jaw_offset_deg: float, ik_tol: float):
    """Continuous IK; the previous solution seeds the next. 연속 IK, 이전 해가 다음 시드."""
    import mujoco
    seed = np.asarray(seed_q, dtype=float).copy()
    out = []
    for i, T in enumerate(poses):
        pos = T[:3, 3]
        R = apply_jaw_offset(T[:3, :3], jaw_offset_deg)
        why, q = None, None
        try:
            q = env.ik(pos, R, seed=seed, strict=False)
        except Exception as exc:                       # noqa: BLE001 — 사유를 남긴다
            why = f"ik_exception:{type(exc).__name__}"
        e_pos = float("nan")
        if q is not None:
            env.ik_data.qpos[env.qids] = q
            mujoco.mj_forward(env.model, env.ik_data)
            Tc = env.tcp(env.ik_data)
            e_pos = float(np.linalg.norm(Tc[:3, 3] - pos))
            if e_pos > ik_tol:
                why = f"position_residual {e_pos * 1000:.2f}mm > {ik_tol * 1000:.1f}mm"
        out.append({"index": i, "q": None if q is None else [float(v) for v in q],
                    "position_residual_m": e_pos, "reject_reason": why})
        if q is not None and why is None:
            seed = np.asarray(q, float)
    return out


# ---------------------------------------------------------------- 자체검증

def selftest() -> int:
    """Known-answer rows, including deliberately broken input. 정답 아는 행 + 손상 입력."""
    rng = np.random.default_rng(0)
    ok = 0
    total = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok, total
        total += 1
        ok += bool(cond)
        print(f"[{total}] {name:<46} {'OK' if cond else '!! 실패'}  {detail}")

    # [1] rot6d 왕복
    errs = []
    for _ in range(2000):
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        r0, r1 = matrix_to_rot6d(Q)
        errs.append(np.abs(rot6d_to_matrix(r0, r1) - Q).max())
    check("rot6d 왕복 2000회", max(errs) < 1e-12, f"최대오차 {max(errs):.2e}")

    # [2] 항등 액션 -> T 불변
    T_cur = np.eye(4)
    T_cur[:3, 3] = [0.4, 0.0, 0.05]
    ident = np.zeros((8, ACTION_DIM))
    ident[:, 3], ident[:, 7] = 1.0, 1.0        # r0=(1,0,0) r1=(0,1,0) = 단위행렬
    ident[:, 9] = 0.05
    poses, gaps, _ = decode_chunk(ident, T_cur)
    check("항등 액션 -> T_next == T_cur",
          max(np.abs(p - T_cur).max() for p in poses) < 1e-12)

    # [3] 순수 병진 +10mm (x)
    a3 = ident.copy()
    a3[:, 0] = 0.010
    poses, _, _ = decode_chunk(a3, T_cur)
    d = poses[0][:3, 3] - T_cur[:3, 3]
    check("순수 병진 +10mm", abs(d[0] - 0.010) < 1e-12 and np.linalg.norm(d[1:]) < 1e-12,
          f"이동 {np.round(d * 1000, 6)} mm")

    # [4] 곱 순서 판별력.
    #     ⚠️ T_cur 의 회전이 단위행렬이면 순수 병진은 순서와 무관하게 교환된다.
    #     초판이 그 경우로 검사해 **판별력 0 인 검사**가 됐고 자체검증이 잡았다.
    #     반드시 회전이 있는 T_cur 로 잰다.
    th = np.deg2rad(90.0)
    T_rot = np.eye(4)
    T_rot[:3, :3] = np.array([[np.cos(th), -np.sin(th), 0.0],
                              [np.sin(th), np.cos(th), 0.0],
                              [0.0, 0.0, 1.0]])
    T_rot[:3, 3] = [0.4, 0.0, 0.05]
    A = row_to_T(a3[0])
    T_right = compose(T_rot, A)
    T_wrong = A @ T_rot
    gap_mm = np.abs(T_right[:3, 3] - T_wrong[:3, 3]).max() * 1000
    check("곱 순서 판별력 (회전 있는 T_cur)", gap_mm > 1e-3, f"차이 {gap_mm:.3f} mm")

    # [4b] 회전 없는 T_cur 에서는 두 순서가 같다 — 이 검사가 언제 눈머는지 명시
    A_t = row_to_T(a3[0])
    check("회전 0 이면 곱 순서가 안 갈린다 (검사의 맹점)",
          np.abs(compose(T_cur, A_t) - A_t @ T_cur).max() < 1e-12)

    # [5] 손상 입력 — 직교성이 깨진 rot6d 를 잔차가 잡는가
    bad = ident[0].copy()
    bad[3:6] = [1.0, 0.0, 0.0]
    bad[6:9] = [0.9, 0.1, 0.0]                 # r0 과 거의 평행
    check("손상 rot6d 잔차 감지", rot6d_residual(bad[3:6], bad[6:9]) > 0.1,
          f"잔차 {rot6d_residual(bad[3:6], bad[6:9]):.3f}")

    # [5b] 완전 평행이면 예외
    par = ident[0].copy()
    par[3:6] = [1.0, 0.0, 0.0]
    par[6:9] = [2.0, 0.0, 0.0]
    try:
        row_to_T(par)
        raised = False
    except ValueError:
        raised = True
    check("평행 rot6d -> 예외", raised)

    # [6] gap 단위 사고 — mm 가 들어오면 거부
    mm = ident.copy()
    mm[:, 9] = 64.23                            # 0.06423 m 를 mm 로 잘못 넣은 모양
    try:
        decode_chunk(mm, T_cur)
        raised = False
    except ValueError:
        raised = True
    check("gap 64.23 (mm 유입) -> 거부", raised)

    # [6b] 정상 gap 은 통과
    try:
        decode_chunk(ident, T_cur)
        passed = True
    except ValueError:
        passed = False
    check("gap 0.05 m 통과", passed)

    # [7] 실행 구간 — 8 중 [1,5) = 4개
    poses, gaps, diag = decode_chunk(ident, T_cur)
    check("exec_slice [1,5) -> 4개", diag["emitted"] == 4 and len(gaps) == 4,
          f"{diag['emitted']} / horizon {diag['horizon']}")

    # [7b] 범위 밖 슬라이스는 거부
    try:
        decode_chunk(ident, T_cur, exec_slice=(1, 99))
        raised = False
    except ValueError:
        raised = True
    check("exec_slice 범위 밖 -> 거부", raised)

    # [8] jaw 오프셋 0 은 항등, 92.79 는 아니다 (판별력)
    R = np.eye(3)
    check("jaw 0도 항등 / 92.79도 변화",
          np.allclose(apply_jaw_offset(R, 0.0), R)
          and not np.allclose(apply_jaw_offset(R, 92.79), R))

    # [9] TCP 게이트 — 레거시 모델이면 반드시 죽어야 한다
    legacy = {"z_local_m": -0.080, "matched": "legacy_so101.yaml", "is_ver1": False}
    try:
        tcp_gate(legacy, None)
        died = False
    except SystemExit:
        died = True
    check("레거시 파지점 -> 게이트가 죽인다", died)

    # [9b] ver1 이면 통과
    ver1 = {"z_local_m": -0.158118819, "matched": "ver1_grasp_so101_ver1.yaml", "is_ver1": True}
    try:
        tcp_gate(ver1, None)
        passed = True
    except SystemExit:
        passed = False
    check("ver1 파지점 -> 통과", passed)

    # [9c] 어느 후보와도 안 맞으면 죽는다
    none_m = {"z_local_m": -0.30, "matched": None, "is_ver1": False}
    try:
        tcp_gate(none_m, None)
        died = False
    except SystemExit:
        died = True
    check("미상 파지점 -> 게이트가 죽인다", died)

    print(f"\n자체검증 {ok} / {total}")
    return 0 if ok == total else 1


# ---------------------------------------------------------------- main

def load_action(path: str) -> np.ndarray:
    p = Path(path).expanduser()
    if p.suffix == ".npy":
        a = np.load(p)
    else:
        a = np.asarray(json.loads(p.read_text(encoding="utf-8")), dtype=float)
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--task", help="handoff task yaml (모델·홈 pose 를 여기서 얻는다)")
    ap.add_argument("--probe-tcp", action="store_true", help="파지점만 재고 끝")
    ap.add_argument("--action", help="(8,10) 액션 청크 .npy 또는 .json")
    ap.add_argument("--from-home", action="store_true", help="T_cur 를 태스크 홈 pose 로")
    ap.add_argument("--from-q", help="현재 관절각 5개 rad, 콤마 구분. T_cur 를 FK 로 구한다")
    ap.add_argument("--exec-slice", default="1:5", help="기본 1:5 (evaluate.py 와 동일)")
    ap.add_argument("--jaw-offset-deg", type=float, default=0.0,
                    help="ver1 jaw 보정. 명시 안 하면 0 이고 리포트에 그렇게 적힌다")
    ap.add_argument("--tcp-offset-m", type=float, default=None,
                    help="파지점 게이트를 건너뛴다. 명시한 사람 책임")
    ap.add_argument("--ik-tol", type=float, default=0.008, help="위치 잔차 허용치 m")
    ap.add_argument("--out", default=None, help="전체 리포트 JSON")
    ap.add_argument("--out-replay", default=None,
                    help="replay_trajectory.py --trajectory 가 먹는 [[q1..q5], ...]")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.task:
        ap.error("--task 가 필요하다 (또는 --selftest)")

    from simulation.env import PickEnv                      # handoff 패키지
    env = PickEnv(task=a.task)
    probe = probe_tcp_offset(env)

    if a.probe_tcp:
        print(json.dumps(probe, ensure_ascii=False, indent=2))
        return

    tcp_gate(probe, a.tcp_offset_m)

    if not a.action:
        ap.error("--action 이 필요하다")
    action = load_action(a.action)
    lo, hi = (int(v) for v in a.exec_slice.split(":"))

    if a.from_q:
        import mujoco
        q0 = np.array([float(v) for v in a.from_q.split(",")], dtype=float)
        if q0.size != len(env.qids):
            ap.error(f"--from-q 가 {q0.size}개다. {len(env.qids)}개여야 한다")
        env.ik_data.qpos[env.qids] = q0
        mujoco.mj_forward(env.model, env.ik_data)
        T_cur = env.tcp(env.ik_data).copy()
        seed_q = q0
    elif a.from_home:
        import mujoco
        q0 = np.asarray(env.home_q, dtype=float)
        env.ik_data.qpos[env.qids] = q0
        mujoco.mj_forward(env.model, env.ik_data)
        T_cur = env.tcp(env.ik_data).copy()
        seed_q = q0
    else:
        ap.error("--from-home 또는 --from-q 중 하나가 필요하다")

    poses, gaps, diag = decode_chunk(action, T_cur, exec_slice=(lo, hi))
    sol = solve_waypoints(env, poses, seed_q, a.jaw_offset_deg, a.ik_tol)

    good = [s for s in sol if s["q"] is not None and s["reject_reason"] is None]
    print(f"\nIK 성공 {len(good)} / 요청 {len(sol)}  (청크 horizon {diag['horizon']}, "
          f"실행구간 [{lo},{hi}))")
    for s in sol:
        mark = "OK" if s["reject_reason"] is None and s["q"] is not None else "거부"
        print(f"  [{s['index']}] {mark:<4} 잔차 {s['position_residual_m'] * 1000:7.3f} mm"
              f"  {s['reject_reason'] or ''}")

    if a.jaw_offset_deg == 0.0:
        print("\n⚠️ --jaw-offset-deg 미지정 → 0 을 썼다. ver1 jaw 는 레거시 대비 "
              "+92.79도다 (grasp_so101_ver1.yaml). 트랙 A 확인 전까지 값을 넣지 않는다.")

    report = {
        "tcp_probe": probe,
        "chunk": diag,
        "T_cur": T_cur.tolist(),
        "jaw_offset_deg": a.jaw_offset_deg,
        "ik_tolerance_m": a.ik_tol,
        "waypoints": sol,
        "gripper_m": gaps,
        "ik_success": len(good),
        "ik_requested": len(sol),
        "conditions": {
            "task": a.task,
            "action_source": a.action,
            "anchor": "home" if a.from_home else f"q={a.from_q}",
            "exec_slice": [lo, hi],
            "rot6d": "first two ROWS",
            "compose": "T_next = T_cur @ A_relative",
        },
    }
    if a.out:
        Path(a.out).expanduser().write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n리포트 -> {a.out}")

    if a.out_replay:
        if len(good) != len(sol):
            raise SystemExit(
                f"!! IK 가 {len(sol) - len(good)}개 거부됐다. 부분 궤적을 실물에 내보내지 않는다.")
        Path(a.out_replay).expanduser().write_text(
            json.dumps([s["q"] for s in good]), encoding="utf-8")
        print(f"재생용 궤적 -> {a.out_replay}  ({len(good)} 웨이포인트)")
        print("⚠️ replay_trajectory.py 의 --gripper 는 궤적 전체에 상수 하나다. "
              f"웨이포인트별 개구는 {[round(g, 4) for g in gaps]} 이고 "
              "현재 재생 도구는 이를 못 따라간다.")


if __name__ == "__main__":
    main()
