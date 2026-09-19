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
DT_S = 0.1          # 액션 점 간격 [s]. 10Hz 공칭 — v10 원본 간격은 균일하지 않다
ROT6D_ROWS = True            # 첫 두 "행". 열이 아니다
GAP_MIN_M, GAP_MAX_M = 0.0, 0.09
GAP_REJECT_ABOVE_M = 0.15    # 단위 사고 감지: mm 가 들어오면 64.23 등으로 즉시 걸린다
EXEC_SLICE = (1, 5)          # evaluate.py 와 동일 (공식 UMI 경로)

# 같은 청크라도 출처에 따라 실행 구간이 다르다 (황도경 확인, 2026-09-19).
# 공식 UMI 정책 출력은 index 1..4, v10 직접 출력은 8개 전부가 미래 목표라 0..3.
# **어느 경로인지 안 적으면 두 경로가 같은 모양으로 보인다.** 그래서 명시를 강제한다.
SOURCE_CONTRACTS = {
    "official_umi": (1, 5),      # f0918_*_B 등 공식 UMI diffusion 출력 — 이번 실물 경로
    "v10_direct": (0, 4),        # 트랙 A v10 직접 출력
}

# 파지점 기준 — 2026-09-19 URDF 실측으로 확정 🟢
#
# `so101_ver1.urdf`(AI/configs/real) 와 `so101_phone_holder.urdf`(handoff 시뮬) 둘 다
# wrist_roll_link -> gripper_tcp 가 소수 9자리까지 **동일**하다.
#     ver1          t = (2.1e-07, -5.64e-07, -0.158118819)  |t| = 0.158118819 m
#     phone_holder  t = (2.1e-07, -5.64e-07, -0.158118819)  |t| = 0.158118819 m
# 즉 handoff 시뮬 모델은 이미 ver1 기하다.
#
# 78.1mm 는 **진짜 기하 차이다** (2026-09-19 대조 🟢).
#   third_party MJCF `so101_new_calib.xml:100` 의 body "gripper" 는
#     pos (5.55112e-17, -0.0611, 0.0181)  quat (0.0172091,-0.0172091,0.706897,0.706897)
#   URDF joint `wrist_roll` 은
#     xyz (0, -0.0611, 0.0181)            rpy (1.5708, 0.04868, 3.14159)
#   병진 동일, 회전 상대각 **0.0003도**. 같은 프레임이다.
#   (판별력 확인: yaw 를 180도 틀리게 넣으면 179.9998도로 갈린다)
#   따라서 so101.yaml 의 -0.080(스톡 SO-101 그리퍼, 패드 중심, 0827 실측)과
#   ver1 의 -0.158118819 는 같은 자로 잰 수이고 **78.118819mm 차이가 실재한다.**
#   도경 회신 "78mm" 와 일치.
#
#   ⚠️ 어시스턴트 정정 이력: 이 건을 두 번 뒤집었다. (1) 78.1mm 를 프레임 차이라고
#      잘못 철회했고 (2) 회전까지 대조해서 원래 경고가 맞았음을 확인했다. 덮지 않는다.
TCP_REFERENCE_BODY = "wrist_roll_link"
TCP_VER1_Z_M = -0.158118819
WRIST_ROLL_MAX_RAD = 1.0471975512   # +60도. 실물 재확인 전 상한 (황도경 2026-09-19).
                                    # 브래킷 실측은 +64.86도지만 마진을 두고 60으로 막는다.
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

JAW_LEGACY_OFFSET_DEG = 92.789      # 레거시 +X jaw 기준 ver1 jaw 까지의 각
JAW_MATCH_TOL_DEG = 5.0


def probe_jaw_axis(env) -> dict:
    """Measure the gripper's opening direction in the TCP frame, from the model.
    그리퍼가 열리는 방향을 TCP 프레임에서 모델로부터 직접 잰다.

    Why this exists / 왜 필요한가 (황도경 지적, 2026-09-19)
    -----------------------------------------------------
    ver1 `gripper_tcp` 축을 목표로 그대로 쓰면 **jaw 방향이 이미 반영돼 있다.**
    그 위에 +92.79도를 또 넣으면 **이중 보정**이라 90도 넘게 틀어진다.
    반대로 레거시 +X jaw 프레임이면 한 번은 넣어야 한다.

    두 경우가 코드에서는 똑같이 생겼다 — 그래서 **재서 가른다.**
    prismatic 그리퍼 관절의 축이 곧 턱이 열리는 방향이다. 그걸 TCP 프레임으로
    옮겨 TCP 의 +X 와 이루는 각을 본다.
    """
    import mujoco
    m = env.model
    mujoco.mj_forward(m, env.data)
    d = env.data

    jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    slides = [i for i, n in enumerate(jnames)
              if n and "grip" in n.lower() and m.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE]
    if not slides:
        raise SystemExit(
            "!! prismatic 그리퍼 관절을 못 찾았다. jaw 축을 잴 수 없다.\n"
            f"   모델 관절 {len(jnames)}개: {jnames}\n"
            "   이름 규칙이 바뀌었으면 여기를 고쳐라. 못 재면 보정값을 추측하지 않는다.")

    jid = slides[0]
    bid = int(m.jnt_bodyid[jid])
    axis_world = d.xmat[bid].reshape(3, 3) @ np.asarray(m.jnt_axis[jid], dtype=np.float64)
    n = np.linalg.norm(axis_world)
    if n < 1e-12:
        raise SystemExit("!! 그리퍼 관절 축의 길이가 0이다. 모델이 이상하다")
    axis_world /= n

    R_tcp = d.site("tcp").xmat.reshape(3, 3)
    axis_tcp = R_tcp.T @ axis_world
    # 턱은 양방향이라 부호가 의미 없다. 0~90도로 접는다.
    c = abs(float(np.clip(axis_tcp[0], -1.0, 1.0)))     # TCP +X 와의 |cos|
    angle_deg = float(np.degrees(np.arccos(c)))

    # 세 축을 전부 돌려준다 (황도경 요청 2026-09-19).
    # jaw 축만 맞고 approach·up 이 틀어져 있으면 "맞다" 로 보이면서 실제로는 돌아간다.
    # TCP 프레임 기준: x = 첫 열, y = 둘째 열, z = 셋째 열.
    axes = {"tcp_x_in_world": [float(v) for v in R_tcp[:, 0]],
            "tcp_y_in_world": [float(v) for v in R_tcp[:, 1]],
            "tcp_z_in_world": [float(v) for v in R_tcp[:, 2]]}
    # jaw 가 TCP 의 어느 축에 가장 가까운가 — 이름을 붙여준다.
    nearest = int(np.argmax(np.abs(axis_tcp)))
    return {
        "joint": jnames[jid],
        "n_gripper_slide_joints": len(slides),
        "n_joints": len(jnames),
        "jaw_axis_in_tcp": [float(v) for v in axis_tcp],
        "jaw_nearest_tcp_axis": "xyz"[nearest],
        "angle_from_tcp_x_deg": angle_deg,
        "angle_from_tcp_y_deg": float(np.degrees(np.arccos(
            abs(float(np.clip(axis_tcp[1], -1.0, 1.0)))))),
        "angle_from_tcp_z_deg": float(np.degrees(np.arccos(
            abs(float(np.clip(axis_tcp[2], -1.0, 1.0)))))),
        "tcp_axes_in_world": axes,
        "tolerance_deg": JAW_MATCH_TOL_DEG,
    }


def jaw_gate(probe: dict, requested_deg: float) -> float:
    """Decide the jaw correction from the measurement and refuse double correction.
    보정값을 계측으로 정하고, 이중 보정을 거부한다.

    돌려주는 값이 실제로 적용할 각이다. 사용자가 지정한 값이 계측과 어긋나면 죽는다."""
    a = probe["angle_from_tcp_x_deg"]
    tol = probe["tolerance_deg"]
    print(f"[jaw] 관절 {probe['joint']} (그리퍼 slide {probe['n_gripper_slide_joints']} / "
          f"전체 관절 {probe['n_joints']})")
    print(f"      jaw 축(TCP 프레임) {np.round(probe['jaw_axis_in_tcp'], 6)}  "
          f"TCP +X 와 {a:.3f}도")

    if a <= tol:
        need = JAW_LEGACY_OFFSET_DEG
        why = (f"jaw 축이 TCP +X 와 {a:.2f}도 — **레거시 +X 프레임**이다. "
               f"{JAW_LEGACY_OFFSET_DEG}도를 한 번 넣어야 한다")
    elif abs(a - JAW_LEGACY_OFFSET_DEG) <= tol:
        need = 0.0
        why = (f"jaw 축이 TCP +X 와 {a:.2f}도 — **ver1 프레임**이다. "
               "이미 반영돼 있으므로 추가 회전 0도")
    else:
        raise SystemExit(
            f"!! jaw 축이 TCP +X 와 {a:.2f}도다. 레거시(0도)도 ver1"
            f"({JAW_LEGACY_OFFSET_DEG}도)도 아니다 (허용 ±{tol}도).\n"
            "   보정값을 추측하지 않는다. 모델·계약을 먼저 확인하라.")

    print(f"      → {why}")
    if abs(requested_deg - need) > 1e-9:
        raise SystemExit(
            f"!! --jaw-offset-deg {requested_deg} 는 계측과 어긋난다. 필요한 값은 "
            f"{need} 도다.\n"
            "   이중 보정하면 90도 넘게 틀어지고, 파지 여유 ±25mm 로는 확정적으로 빗나간다.\n"
            f"   맞다고 확신하면 --jaw-offset-deg {need} 로 명시하라.")
    return need


def probe_tcp_offset(env) -> dict:
    """Measure site `tcp` in the reference link frame, from the model itself.
    site `tcp` 를 기준 링크 프레임에서 모델로부터 직접 잰다. body 이름을 추측하지 않는다."""
    import mujoco
    m = env.model
    mujoco.mj_forward(m, env.data)
    d = env.data

    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    sid = m.site("tcp").id
    owner = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.site_bodyid[sid]))

    rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TCP_REFERENCE_BODY)
    if rid < 0:
        raise SystemExit(
            f"!! 기준 body '{TCP_REFERENCE_BODY}' 가 모델에 없다.\n"
            f"   모델의 body {len(bodies)}개: {bodies}\n"
            "   기준 링크 이름이 바뀌었다. TCP_REFERENCE_BODY 를 고쳐라.")

    Rr = d.xmat[rid].reshape(3, 3)
    pr = d.xpos[rid]
    local = Rr.T @ (d.site("tcp").xpos - pr)
    dz = abs(float(local[2]) - TCP_VER1_Z_M)
    return {
        "reference_body": TCP_REFERENCE_BODY,
        "site_owner_body": owner,
        "site_tcp_local_m": [float(v) for v in local],
        "z_local_m": float(local[2]),
        "norm_m": float(np.linalg.norm(local)),
        "ver1_expected_z_m": TCP_VER1_Z_M,
        "abs_diff_m": dz,
        "tolerance_m": TCP_MATCH_TOL_M,
        "is_ver1": bool(dz <= TCP_MATCH_TOL_M),
        "n_bodies": len(bodies),
    }


def tcp_gate(probe: dict, override: float | None) -> None:
    """Refuse to emit a trajectory when the model's pinch point is not ver1's.
    모델의 파지점이 ver1 이 아니면 궤적을 내놓지 않는다."""
    if override is not None:
        print(f"[TCP] --tcp-offset-m {override} 명시됨 — 게이트 건너뜀 (사용자 책임)")
        return
    print(f"[TCP] 기준 {probe['reference_body']} · site 소유 body {probe['site_owner_body']} "
          f"· 모델 body {probe['n_bodies']}개")
    print(f"      site tcp 로컬 = {np.round(probe['site_tcp_local_m'], 9)}  "
          f"|t| = {probe['norm_m']:.9f} m")
    print(f"      ver1 기대 z   = {probe['ver1_expected_z_m']:+.9f} m  "
          f"차이 {probe['abs_diff_m'] * 1000:.3f} mm")
    if not probe["is_ver1"]:
        raise SystemExit(
            f"!! 파지점이 ver1({TCP_VER1_Z_M:+.9f} m) 과 "
            f"{probe['abs_diff_m'] * 1000:.1f}mm 어긋난다. 궤적을 내지 않는다.\n"
            "   파지 여유는 ±25mm = (개구 90 − 물체 41)/2 뿐이고 이런 상수 편향은\n"
            "   학습이 지우지 못한다. 모델을 바꾸거나 --tcp-offset-m 을 명시하라.")
    print("[TCP] ver1 파지점 확인. 통과")


# ---------------------------------------------------------------- env 생성

def make_env(PickEnv, task_path: str):
    """Construct PickEnv against its ACTUAL signature, not a remembered one.
    기억이 아니라 **실제 시그니처**에 맞춰 PickEnv 를 만든다.

    2026-09-19: `PickEnv(task=...)` 로 불러 TypeError 가 났다. 소스는
    `(camera_path, object_kind, task_path, task_config)` 다. 인자를 소스로 대조하지
    않고 쓴 내 잘못이고, 같은 실수를 반복하지 않도록 여기서 검사한다."""
    import inspect
    params = list(inspect.signature(PickEnv.__init__).parameters)
    if "task_path" not in params:
        raise SystemExit(
            f"!! PickEnv 가 task_path 를 안 받는다. 실제 인자: {params}\n"
            "   handoff 버전이 바뀌었다. 소스를 보고 이 호출을 고쳐라.")
    return PickEnv(task_path=task_path)


# ---------------------------------------------------------------- IK

def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle between two rotations [deg]. 두 회전 사이의 각."""
    c = (float(np.trace(np.asarray(Ra).T @ np.asarray(Rb))) - 1.0) / 2.0
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


WRIST_ROLL_SEEDS_RAD = (-0.6, -0.3, 0.0, 0.3, 0.6)
"""Seeds to try for the redundant wrist roll. 여유 자유도인 손목 회전의 시드 후보.

왜 여러 개인가 (황도경 요청 2026-09-19): 5축에서 손목 회전은 측면 파지의 접근축
둘레 자유도에 해당해 해가 여러 개 나올 수 있다. IK 를 한 시드로만 부르면 그중
**턱 방향이 가장 틀어진 해**가 걸릴 수 있고, 그건 위치 잔차로는 안 잡힌다."""


def solve_waypoints(env, poses, seed_q, jaw_offset_deg: float, ik_tol: float,
                    jaw_tol_deg: float = 10.0):
    """Continuous IK; among wrist-roll candidates, keep the smallest jaw error.
    연속 IK. 손목 회전 후보 중 **턱 방향 오차가 가장 작은 해**를 고른다.

    위치 잔차만 보면 턱이 돌아간 해도 통과한다. 목표 자세와의 측지각을 같이 보고,
    그 값을 리포트에 남긴다. 그리고 실물 재확인 전까지 손목 회전은 +60도에서 막는다."""
    import mujoco
    seed = np.asarray(seed_q, dtype=float).copy()
    wr_idx = 4                                  # pan/lift/elbow/wrist_flex/wrist_roll
    out = []
    for i, T in enumerate(poses):
        pos = T[:3, 3]
        R = apply_jaw_offset(T[:3, :3], jaw_offset_deg)

        best = None
        tried = 0
        for wr in (seed[wr_idx], *WRIST_ROLL_SEEDS_RAD):
            s2 = seed.copy()
            s2[wr_idx] = float(wr)
            tried += 1
            try:
                q = env.ik(pos, R, seed=s2, strict=False)
            except Exception:                    # noqa: BLE001 — 후보 하나가 죽어도 계속
                continue
            if q is None:
                continue
            if float(q[wr_idx]) > WRIST_ROLL_MAX_RAD:
                continue                          # 실물 재확인 전 상한
            env.ik_data.qpos[env.qids] = q
            mujoco.mj_forward(env.model, env.ik_data)
            Tc = env.tcp(env.ik_data)
            e_pos = float(np.linalg.norm(Tc[:3, 3] - pos))
            e_rot = geodesic_deg(R, Tc[:3, :3])
            if best is None or e_rot < best["e_rot"]:
                best = {"q": q, "e_pos": e_pos, "e_rot": e_rot}

        why = None
        if best is None:
            why = f"no_ik_solution (후보 {tried}개 전부 실패 또는 wrist_roll 상한 초과)"
            e_pos = e_rot = float("nan")
            q = None
        else:
            q, e_pos, e_rot = best["q"], best["e_pos"], best["e_rot"]
            if e_pos > ik_tol:
                why = f"position_residual {e_pos * 1000:.2f}mm > {ik_tol * 1000:.1f}mm"
            elif e_rot > jaw_tol_deg:
                why = f"jaw_axis_residual {e_rot:.2f}deg > {jaw_tol_deg:.1f}deg"

        out.append({"index": i, "q": None if q is None else [float(v) for v in q],
                    "position_residual_m": e_pos,
                    "rotation_residual_deg": e_rot,
                    "wrist_roll_candidates_tried": tried,
                    "reject_reason": why})
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

    # [9] TCP 게이트 — ver1 이 아니면 반드시 죽어야 한다
    legacy = {"reference_body": TCP_REFERENCE_BODY, "site_owner_body": "x",
              "site_tcp_local_m": [0, 0, -0.080], "z_local_m": -0.080,
              "norm_m": 0.080, "ver1_expected_z_m": TCP_VER1_Z_M,
              "abs_diff_m": abs(-0.080 - TCP_VER1_Z_M), "tolerance_m": TCP_MATCH_TOL_M,
              "is_ver1": False, "n_bodies": 3}
    try:
        tcp_gate(legacy, None)
        died = False
    except SystemExit:
        died = True
    check("ver1 아닌 파지점 -> 게이트가 죽인다", died)

    ver1 = dict(legacy, z_local_m=TCP_VER1_Z_M, site_tcp_local_m=[0, 0, TCP_VER1_Z_M],
                norm_m=abs(TCP_VER1_Z_M), abs_diff_m=0.0, is_ver1=True)
    try:
        tcp_gate(ver1, None)
        passed = True
    except SystemExit:
        passed = False
    check("ver1 파지점 -> 통과", passed)

    # 판별력: 허용치 바로 안/밖이 갈리는가
    edge_in = dict(ver1, z_local_m=TCP_VER1_Z_M + 0.0019,
                   abs_diff_m=0.0019, is_ver1=True)
    edge_out = dict(ver1, z_local_m=TCP_VER1_Z_M + 0.0021,
                    abs_diff_m=0.0021, is_ver1=False)
    try:
        tcp_gate(edge_in, None)
        a_ok = True
    except SystemExit:
        a_ok = False
    try:
        tcp_gate(edge_out, None)
        b_ok = True
    except SystemExit:
        b_ok = False
    check("허용치 1.9mm 통과 / 2.1mm 거부 (판별력)", a_ok and not b_ok)

    # [16~20] jaw 게이트 — 이중 보정을 막는 것이 요점이다 (황도경 2026-09-19)
    ver1_jaw = {"joint": "gripper_right", "n_gripper_slide_joints": 2, "n_joints": 8,
                "jaw_axis_in_tcp": [0.0, 1.0, 0.0],
                "angle_from_tcp_x_deg": 90.0, "tolerance_deg": JAW_MATCH_TOL_DEG}
    legacy_jaw = dict(ver1_jaw, jaw_axis_in_tcp=[1.0, 0.0, 0.0],
                      angle_from_tcp_x_deg=0.0)

    check("ver1 jaw 프레임 -> 추가 회전 0도 (정답 아는 행)",
          jaw_gate(ver1_jaw, 0.0) == 0.0)
    check("레거시 jaw 프레임 -> 92.789도 필요",
          jaw_gate(legacy_jaw, JAW_LEGACY_OFFSET_DEG) == JAW_LEGACY_OFFSET_DEG)
    try:
        jaw_gate(ver1_jaw, JAW_LEGACY_OFFSET_DEG)
        caught = False
    except SystemExit:
        caught = True
    check("판별력: ver1 인데 92.79도를 넣으면 거부 (이중 보정 차단)", caught)
    try:
        jaw_gate(dict(ver1_jaw, angle_from_tcp_x_deg=45.0), 0.0)
        caught2 = False
    except SystemExit:
        caught2 = True
    check("판별력: 둘 다 아닌 각(45도)이면 추측하지 않고 죽는다", caught2)
    check("source_contract 두 경로의 실행 구간이 다르다",
          SOURCE_CONTRACTS["official_umi"] == (1, 5)
          and SOURCE_CONTRACTS["v10_direct"] == (0, 4))

    # [21~24] 손목 회전 상한 · 측지각 (황도경 요청 2026-09-19)
    check("wrist_roll 상한이 +60도 (실물 재확인 전)",
          abs(np.degrees(WRIST_ROLL_MAX_RAD) - 60.0) < 1e-6,
          f"{np.degrees(WRIST_ROLL_MAX_RAD):.4f}도")
    check("측지각: 같은 회전이면 0도 (정답 아는 행)",
          abs(geodesic_deg(np.eye(3), np.eye(3))) < 1e-9)
    Rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    check("측지각: z 90도 회전이면 90도", abs(geodesic_deg(np.eye(3), Rz90) - 90.0) < 1e-9)
    check("판별력: 92.79도 jaw 보정의 측지각이 92.79도",
          abs(geodesic_deg(np.eye(3), apply_jaw_offset(np.eye(3), 92.789))
              - 92.789) < 1e-6)
    check("손목 시드 후보가 여러 개다 (한 시드면 턱 틀어진 해가 걸린다)",
          len(WRIST_ROLL_SEEDS_RAD) >= 3)

    # [26~28] 재생 출력이 gap 을 같이 싣는가 (황도경 지적 2026-09-19)
    fake_q = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.11, 0.21, 0.31, 0.41, 0.51]]
    fake_gap = [0.070, 0.041]
    rows = [[*q, float(g)] for q, g in zip(fake_q, fake_gap)]
    check("재생 행이 6열이다 (관절5 + gap)", all(len(r) == 6 for r in rows))
    check("gap 이 행마다 다르다 (상수 명령이 아니다)",
          rows[0][5] != rows[1][5], f"{rows[0][5]} vs {rows[1][5]}")
    check("판별력: q 만 내보내면 5열이라 gap 이 사라진다",
          len(fake_q[0]) == 5 and 0.041 not in fake_q[0])
    check("점 간격 상수가 10Hz", abs(DT_S - 0.1) < 1e-12)

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
    # 기본값을 주지 않는다 (황도경 지적 2026-09-19): 기본값이 있으면 엄밀히는
    # 필수 입력이 아니고, **안 정한 것과 정한 것이 같은 모양**이 된다.
    ap.add_argument("--source-contract", choices=sorted(SOURCE_CONTRACTS),
                    default=None,
                    help="액션 청크의 출처. 실행 구간이 여기서 정해진다 — "
                         "official_umi 1..4 / v10_direct 0..3 (황도경 확인 2026-09-19). "
                         "**필수다. 기본값 없다**")
    ap.add_argument("--exec-slice", default=None,
                    help="실행 구간을 직접 지정 (예 1:5). 주면 --source-contract 기본값을 덮는다")
    ap.add_argument("--jaw-offset-deg", type=float, default=0.0,
                    help="ver1 jaw 보정. 명시 안 하면 0 이고 리포트에 그렇게 적힌다")
    ap.add_argument("--tcp-offset-m", type=float, default=None,
                    help="파지점 게이트를 건너뛴다. 명시한 사람 책임")
    ap.add_argument("--ik-tol", type=float, default=0.008, help="위치 잔차 허용치 m")
    ap.add_argument("--jaw-tol-deg", type=float, default=10.0,
                    help="목표 자세와의 측지각 허용치 [도]. 위치만 보면 턱이 돌아간 해가 통과한다")
    ap.add_argument("--out", default=None, help="전체 리포트 JSON")
    ap.add_argument("--out-replay", default=None,
                    help="replay_trajectory.py --trajectory 가 먹는 [[q1..q5], ...]")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.task:
        ap.error("--task 가 필요하다 (또는 --selftest)")
    if not a.source_contract:
        ap.error("--source-contract 가 필요하다 (official_umi / v10_direct). "
                 "기본값을 두지 않는다 — 안 정한 것과 정한 것이 같은 모양이 되면 안 된다")

    from simulation.env import PickEnv                      # handoff 패키지
    env = make_env(PickEnv, a.task)
    probe = probe_tcp_offset(env)
    jaw_probe = probe_jaw_axis(env)

    if a.probe_tcp:
        print(json.dumps({"tcp": probe, "jaw": jaw_probe}, ensure_ascii=False, indent=2))
        return

    tcp_gate(probe, a.tcp_offset_m)
    jaw_deg = jaw_gate(jaw_probe, a.jaw_offset_deg)

    if not a.action:
        ap.error("--action 이 필요하다")
    action = load_action(a.action)
    if a.exec_slice:
        lo, hi = (int(v) for v in a.exec_slice.split(":"))
        print(f"[구간] --exec-slice {a.exec_slice} 로 덮어씀 "
              f"(source_contract {a.source_contract} 기본값은 "
              f"{SOURCE_CONTRACTS[a.source_contract]})")
    else:
        lo, hi = SOURCE_CONTRACTS[a.source_contract]
        print(f"[구간] source_contract={a.source_contract} → index {lo}..{hi-1}")

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
    sol = solve_waypoints(env, poses, seed_q, jaw_deg, a.ik_tol, a.jaw_tol_deg)

    good = [s for s in sol if s["q"] is not None and s["reject_reason"] is None]
    print(f"\nIK 성공 {len(good)} / 요청 {len(sol)}  (청크 horizon {diag['horizon']}, "
          f"실행구간 [{lo},{hi}))")
    for s in sol:
        mark = "OK" if s["reject_reason"] is None and s["q"] is not None else "거부"
        print(f"  [{s['index']}] {mark:<4} 잔차 {s['position_residual_m'] * 1000:7.3f} mm"
              f"  {s['reject_reason'] or ''}")

    print(f"\n[jaw] 실제 적용 {jaw_deg} 도 (계측 기반. 추측값 아님)")

    report = {
        "tcp_probe": probe,
        "chunk": diag,
        "T_cur": T_cur.tolist(),
        "jaw_offset_deg": jaw_deg,
        "jaw_probe": jaw_probe,
        "source_contract": a.source_contract,
        "ik_tolerance_m": a.ik_tol,
        "waypoints": sol,
        "gripper_m": gaps,
        "dt_s": DT_S,
        "waypoint_times_s": [round(i * DT_S, 4) for i in range(len(sol))],
        "exec_duration_s": round((hi - lo) * DT_S, 4),
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
        # 웨이포인트별 gap 을 **같은 행에** 싣는다 (황도경 지적 2026-09-19).
        # q 만 내보내면 재생 쪽에서 상수 그리퍼 명령으로 바뀌고, 그건 관통이 아니다.
        # replay_trajectory.parse_waypoints 가 [q1..q5, gap_m] 6폭을 받는다.
        if len(gaps) != len(good):
            raise SystemExit(
                f"!! gap {len(gaps)}개 / 웨이포인트 {len(good)}개 — 개수가 다르다. "
                "행 대응이 깨지면 조용히 어긋난다. 내보내지 않는다")
        rows = [[*s["q"], float(g)] for s, g in zip(good, gaps)]
        Path(a.out_replay).expanduser().write_text(json.dumps(rows), encoding="utf-8")
        print(f"재생용 궤적 -> {a.out_replay}  "
              f"({len(rows)} 웨이포인트 × 6열 = 관절5 + gap_m)")
        print(f"  웨이포인트별 개구 {[round(g, 4) for g in gaps]}")
        print(f"  점 간격 {DT_S:.3f} s (10Hz 공칭) · 실행 구간 {hi - lo}점 "
              f"= {(hi - lo) * DT_S:.2f} s")


if __name__ == "__main__":
    main()
