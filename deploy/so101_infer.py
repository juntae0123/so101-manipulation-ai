"""Policy checkpoint -> executable SO-101 joint trajectory (ROS-free, hardware-free).
정책 체크포인트 출력을 실물이 그대로 먹는 관절각 궤적으로 바꾼다. ROS·드라이버 의존 없음.

계약 출처: ~/handoff/outputs/deploy/so101_pick_v1.manifest.json (f0918_B_42)
기구학 출처: AI/deploy/vendor/shanks_kinematics.py (김현석 handoff 2026-09-18, 무수정)

사용:
    python3 so101_infer.py --selftest
    python3 so101_infer.py --actions chunk.npy --q0 0,-0.5,0.5,0,0 --out-dir out/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import shanks_kinematics as K  # noqa: E402

CONTRACT_VERSION = "so101-policy-v1"

# ── 액션 계약. 출처: so101_pick_v1.manifest.json actionSpec ──────────────────
ACTION_DIM = 10
ACTION_HORIZON = 8
EXEC_SLICE = (1, 5)              # execSlice. cfg 값이 아니라 실행측 선택이다
ROT6D_ROWS = True                # "회전행렬의 첫 두 행. 열이 아니다"
NOMINAL_RATE_HZ = 10.0           # 공칭. 실제 시각은 observation_timestamp 에서 읽는다.
#  ⚠️ 정정 2026-09-21 — "간격 2·3 혼재" 는 틀렸다. 0918 v10 74편·차분 4640개 전부 source_row step=3 으로 확정 (2026-09-21, 황도경).
#     그래도 0.1초를 고정으로 가정하지 않는다: 배치가 바뀌면 다시 재고 쓴다
GAP_RANGE_M = (0.0, 0.09)
ACTION_POSE_REPR = "relative"    # 기본값 'abs'. 안 넘기면 상대를 절대로 읽는다

# ── 안전 한계는 상수로 두지 않는다 (황도경 지적 2026-09-20) ──────────────────
# 초판은 MAX_JOINT_SPEED_RAD_S = 0.3 을 상수로 박았다. **내가 정한 값이고 근거가 없다.**
# 근거 없는 값이 코드에 있으면 "설정을 안 줘도 돌아간다" 가 되어 fail-closed 가 깨진다.
# 이제 전부 SAFETY_PATH 에서만 읽는다. 누락·null·NaN 이면 실행을 거부한다.
SAFETY_PATH = Path(__file__).resolve().parents[1] / "configs" / "real" / "so101_safety.json"
SAFETY_KEYS = (
    "max_joint_speed_rad_s",      # HW 미수령
    "max_step_tick",              # control/shanks/move.py 실측 300 🟢
    "min_joint_margin_deg",       # HW 미수령
    "wrist_roll_min_rad",         # HW 미수령 — URDF 하한은 물리 정지가 아니다 (D-AI-56)
    "wrist_roll_max_rad",         # HW 미수령
    "min_tcp_z_table_mm",
    "ik_tol_mm",
)
TICKS_PER_TURN = 4096
GRIPPER_CLOSED_TICK, GRIPPER_OPEN_TICK, GRIPPER_OPEN_WIDTH_M = 1763, 130, 0.09
WRIST_ROLL_INDEX = 4


class SafetyUnset(RuntimeError):
    """Raised when a safety threshold is missing/null/NaN — never fall back to a guess.
    안전 임계값이 없거나 null·NaN 이면 발생. 추측값으로 대체하지 않는다."""


def load_safety(path: Path | None = None) -> dict:
    """Load safety thresholds. Missing / null / NaN / non-finite -> refuse.
    안전 임계값을 읽는다. 누락·null·NaN·무한대는 전부 거부한다 (fail-closed)."""
    path = Path(path) if path else SAFETY_PATH
    if not path.exists():
        raise SafetyUnset(f"안전 설정 파일이 없다: {path}\n"
                          f"   HW 안전 기준을 받아 채워라. 값을 지어내지 않는다.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    missing, bad = [], []
    out = {}
    for k in SAFETY_KEYS:
        if k not in raw or raw[k] is None:
            missing.append(k)
            continue
        v = raw[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
            bad.append(f"{k}={v!r}")
            continue
        out[k] = float(v)
    if missing or bad:
        raise SafetyUnset(
            f"안전 임계값 {len(out)}/{len(SAFETY_KEYS)} 만 유효하다 — 실행 거부.\n"
            f"   누락·null: {missing or '없음'}\n"
            f"   NaN·비수치: {bad or '없음'}\n"
            f"   파일: {path}")
    if out["wrist_roll_min_rad"] >= out["wrist_roll_max_rad"]:
        raise SafetyUnset("wrist_roll 범위가 뒤집혀 있다 (min >= max)")
    out["max_step_rad"] = out["max_step_tick"] * 2 * math.pi / TICKS_PER_TURN
    return out

# ── jaw 규약. URDF so101_ver1_original.urdf 에서 직접 계산 (스윕 불필요) ────────
# gripper_right axis = custom_hand_link +X, gripper_tcp_fixed rpy=(pi/2,0,0)
#   -> jaw 개폐축 == gripper_tcp +X, 보정각 0.000000 deg
# manifest 의 92.7889 deg 는 MJCF gripper body 프레임 기준이다. 여기선 쓰지 않는다.
JAW_AXIS_IN_TCP = np.array([1.0, 0.0, 0.0])
JAW_OFFSET_DEG = 0.0


# ── 회전 표현 ────────────────────────────────────────────────────────────────
def rot6d_to_matrix(a6, rows: bool = ROT6D_ROWS) -> np.ndarray:
    """6D rotation -> 3x3. rows=True 면 첫 두 '행', False 면 '열'.
    행/열을 바꿔 읽으면 전치가 나오고 중앙오차 수 mm 가 조용히 남는다."""
    a = np.asarray(a6, float).reshape(2, 3)
    b0 = a[0] / np.linalg.norm(a[0])
    b1 = a[1] - (b0 @ a[1]) * b0
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=0 if rows else 1)


def matrix_to_rot6d(R, rows: bool = ROT6D_ROWS) -> np.ndarray:
    R = np.asarray(R, float)
    return (R[:2, :] if rows else R[:, :2].T).reshape(6).copy()


def action_to_T(a10) -> tuple[np.ndarray, float]:
    """액션 1행 -> (상대 4x4, gap[m])."""
    a = np.asarray(a10, float)
    if a.shape != (ACTION_DIM,):
        raise ValueError(f"액션 차원이 {a.shape} 다. ({ACTION_DIM},) 여야 한다")
    gap = float(a[9])
    if not (GAP_RANGE_M[0] - 1e-9 <= gap <= GAP_RANGE_M[1] + 1e-9):
        raise ValueError(f"gap {gap:.4f} m 가 범위 {GAP_RANGE_M} 밖이다")
    T = np.eye(4)
    T[:3, :3] = rot6d_to_matrix(a[3:9])
    T[:3, 3] = a[0:3]
    return T, gap


def unroll(T_start, chunks, exec_slice=EXEC_SLICE) -> tuple[list, list]:
    """상대 액션 청크들을 절대 TCP 궤적으로 편다.

    ⚠️ 2026-09-20 정정 (D-AI-80). 초판은 청크 **안에서** `T = T @ A` 로 누적했다.
    v10 실데이터 20편 · 비교 5648건에서 앵커 `T0 @ A[k]` 만 오차 0.0000mm 였다:
        누적 T0@A0@..@Ak   위치 중앙  59.7291 mm
        앵커 T0@Ak         위치 중앙   0.0000 mm   ← 정답
        초판 공식           위치 중앙  44.7580 mm (k=1 0.0 / 2 40.1 / 3 98.3 / 4 171.3)
    액션 한 행은 전부 **청크 시작 pose 기준**이다. 청크 안에서 앵커는 바뀌지 않는다.
    청크 사이에서만 앵커가 마지막 실행점으로 넘어간다 (실물에서는 재관측 pose).
    근거: AI/tools/probe_chunk_anchor.py · out/anchor_0920.json
    """
    lo, hi = exec_slice
    T, poses, gaps = np.asarray(T_start, float).copy(), [], []
    for ch in chunks:
        ch = np.asarray(ch, float)
        if ch.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"청크 shape {ch.shape}, 기대 ({ACTION_HORIZON},{ACTION_DIM})")
        T0 = T.copy()                    # 이 청크의 앵커. 청크 안에서 고정이다
        for row in ch[lo:hi]:
            A, gap = action_to_T(row)
            T = T0 @ A
            poses.append(T.copy())
            gaps.append(gap)
    return poses, gaps


# ── 기구학 ───────────────────────────────────────────────────────────────────
def ik_chain(poses, q0, constrain_jaw: bool = True):
    """TCP 4x4 열 -> 관절각. 직전 해를 시드로 이어 푼다."""
    qs, res, ok_all = [], [], True
    q = np.asarray(q0, float)
    for T in poses:
        approach = T[:3, 2]
        finger = T[:3, 0] if constrain_jaw else None
        q, r, ok = K.ik(T[:3, 3], approach=approach, finger=finger, q0=q)
        qs.append(np.asarray(q, float).copy())
        res.append(float(r))
        ok_all = ok_all and bool(ok)
    return np.array(qs), np.array(res), ok_all


def gap_to_tick(gap_m: float) -> int:
    r = gap_m / GRIPPER_OPEN_WIDTH_M
    return int(round(GRIPPER_CLOSED_TICK + (GRIPPER_OPEN_TICK - GRIPPER_CLOSED_TICK) * r))


# ── 프리플라이트 ─────────────────────────────────────────────────────────────
def preflight(qs, gaps, dt_s, residual_mm=None, safety: dict | None = None) -> dict:
    """실행 전 검사. 모든 항목이 모수(검사 수)를 같이 낸다.
    safety 는 필수다. 없으면 SafetyUnset 으로 죽는다 — 기본값으로 넘어가지 않는다."""
    if safety is None:
        raise SafetyUnset("preflight 에 안전 설정이 안 넘어왔다. 기본값으로 대체하지 않는다")
    qs = np.asarray(qs, float)
    n = len(qs)
    if n and qs.shape[1] != len(K.RAD_LIMITS):
        raise ValueError(f"관절 수가 {qs.shape[1]} 이다. {len(K.RAD_LIMITS)} 이어야 한다")
    lo, hi = np.array(K.RAD_LIMITS).T
    MAX_JOINT_SPEED_RAD_S = safety["max_joint_speed_rad_s"]
    MAX_STEP_RAD = safety["max_step_rad"]
    MIN_TCP_Z_TABLE_MM = safety["min_tcp_z_table_mm"]
    IK_TOL_MM = safety["ik_tol_mm"]
    rep: dict = {"n_waypoints": n, "dt_s": dt_s, "safety": dict(safety), "checks": {}, "fail": []}

    bad = [i for i in range(n) if np.any(qs[i] < lo - 1e-9) or np.any(qs[i] > hi + 1e-9)]
    rep["checks"]["joint_limit"] = {"violations": len(bad), "checked": n, "idx": bad[:10]}

    steps = np.abs(np.diff(qs, axis=0)) if n > 1 else np.zeros((0, 5))
    smax = float(steps.max()) if steps.size else 0.0
    nstep = int((steps > MAX_STEP_RAD + 1e-9).sum())
    rep["checks"]["max_step"] = {"violations": nstep, "checked": int(steps.size),
                                 "max_rad": smax, "limit_rad": MAX_STEP_RAD}

    speed = steps / dt_s if steps.size else np.zeros((0, 5))
    vmax = float(speed.max()) if speed.size else 0.0
    nspd = int((speed > MAX_JOINT_SPEED_RAD_S + 1e-9).sum())
    need_dt = (smax / MAX_JOINT_SPEED_RAD_S) if smax > 0 else dt_s
    rep["checks"]["speed"] = {"violations": nspd, "checked": int(speed.size),
                              "max_rad_s": vmax, "limit_rad_s": MAX_JOINT_SPEED_RAD_S,
                              "required_dt_s": round(need_dt, 4),
                              "slowdown_x": round(need_dt / dt_s, 2) if dt_s > 0 else None}

    z = np.array([K.table_from_base(K.fk(q)[:3, 3])[2] for q in qs])
    nz = int((z < MIN_TCP_Z_TABLE_MM - 1e-9).sum())
    rep["checks"]["min_z"] = {"violations": nz, "checked": n,
                              "min_mm": float(z.min()) if n else None,
                              "limit_mm": MIN_TCP_Z_TABLE_MM}

    g = np.asarray(gaps, float)
    ng = int(((g < GAP_RANGE_M[0] - 1e-9) | (g > GAP_RANGE_M[1] + 1e-9)).sum())
    rep["checks"]["gap_range"] = {"violations": ng, "checked": len(g)}

    # wrist_roll 실물 사용 범위. URDF 한계 안이어도 물리 정지·브래킷 간섭이 먼저 온다.
    # 2026-09-19 실물 로그에서 337틱(-150.4도)이 나왔는데 URDF 범위 안이라
    # joint_limit 검사에 안 걸렸다. 상한만 걸면 음수 쪽이 그대로 통과한다 — 양쪽을 본다.
    wr = qs[:, WRIST_ROLL_INDEX] if n else np.zeros(0)
    wlo, whi = safety["wrist_roll_min_rad"], safety["wrist_roll_max_rad"]
    nwr = int(((wr < wlo - 1e-9) | (wr > whi + 1e-9)).sum())
    rep["checks"]["wrist_roll_range"] = {
        "violations": nwr, "checked": int(len(wr)),
        "limit_rad": [wlo, whi], "limit_deg": [round(math.degrees(wlo), 3), round(math.degrees(whi), 3)],
        "min_rad": float(wr.min()) if n else None, "max_rad": float(wr.max()) if n else None}

    # 관절 한계까지 남은 여유. 통과/불통과만으로는 아슬아슬한 궤적과 여유 있는 궤적이
    # 같은 출력이다. 2026-09-19 실물에서 wrist_flex 가 시작부터 15.6도 앞이었다.
    if n:
        M = np.degrees(np.minimum(qs - lo, hi - qs))
        per_joint = M.min(axis=0)
        j = int(np.argmin(per_joint))
        worst = float(per_joint[j])
    else:
        per_joint, j, worst = np.zeros(0), None, None
    thr = safety["min_joint_margin_deg"]
    rep["checks"]["joint_margin"] = {
        "violations": int((M.min(axis=1) < thr).sum()) if n else 0,
        "checked": n, "threshold_deg": thr,
        "per_joint_min_deg": [round(float(v), 3) for v in per_joint] if n else None,
        "trajectory_min_deg": round(worst, 3) if worst is not None else None,
        "tightest_joint_index": j}

    if residual_mm is not None:
        r = np.asarray(residual_mm, float)
        rep["checks"]["ik_residual"] = {"violations": int((r > IK_TOL_MM).sum()),
                                        "checked": len(r), "max_mm": float(r.max())}

    for name, c in rep["checks"].items():
        if c["checked"] == 0:
            rep["fail"].append(f"{name}: 검사 0건 — 범위가 비었다")
        elif c["violations"] > 0:
            rep["fail"].append(f"{name}: {c['violations']}/{c['checked']} 위반")
    rep["ok"] = not rep["fail"]
    return rep


# ── A/B/C 대조군 ─────────────────────────────────────────────────────────────
def make_abc(chunks, T_start, q0, dt_s, safety: dict, seed: int = 42) -> dict:
    """A 정책 · B 정지 · C 셔플. B·C 없이는 '팔이 움직였다'가 증거가 못 된다.
    C 는 '상대 델타의 순서'를 섞는다 — 궤적은 매끄럽고 의미만 사라지므로 실행 가능하다."""
    out = {}
    chunks = [np.asarray(c, float) for c in chunks]

    poses, gaps = unroll(T_start, chunks)
    qs, res, _ = ik_chain(poses, q0)
    out["A_policy"] = (qs, gaps, preflight(qs, gaps, dt_s, res, safety))

    n = len(qs)
    out["B_hold"] = (np.repeat(np.asarray(q0, float)[None, :], n, axis=0),
                     [gaps[0]] * n, None)
    out["B_hold"] = (out["B_hold"][0], out["B_hold"][1],
                     preflight(out["B_hold"][0], out["B_hold"][1], dt_s, None, safety))

    lo, hi = EXEC_SLICE
    rows = np.concatenate([c[lo:hi] for c in chunks], axis=0)
    rows = rows[np.random.default_rng(seed).permutation(len(rows))]
    cs = [rows[i:i + (hi - lo)] for i in range(0, len(rows), hi - lo)]
    pad = [np.concatenate([np.zeros((lo, ACTION_DIM)), c,
                           np.zeros((ACTION_HORIZON - hi, ACTION_DIM))], axis=0) for c in cs
           if len(c) == hi - lo]
    for p in pad:
        p[:, 3:9] = matrix_to_rot6d(np.eye(3))
    p2, g2 = unroll(T_start, pad)
    q2, r2, _ = ik_chain(p2, q0)
    out["C_shuffle"] = (q2, g2, preflight(q2, g2, dt_s, r2, safety))
    return out


def write_trajectory(path, qs, gaps, dt_s, meta=None) -> Path:
    doc = {
        "contractVersion": CONTRACT_VERSION,
        "arm": "so101_ver1",
        "frame": "relative_from_start",
        "units": {"q": "rad", "gap": "m", "t": "s"},
        "joint_names": list(K.JOINT_NAMES),
        "dt_s": dt_s,
        "n_waypoints": len(qs),
        "waypoints": [{"t_s": round(i * dt_s, 6),
                       "q_rad": [round(float(v), 9) for v in q],
                       "gap_m": round(float(g), 6),
                       "gripper_tick": gap_to_tick(float(g))}
                      for i, (q, g) in enumerate(zip(qs, gaps))],
        "meta": meta or {},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


class MockPublisher:
    """Counts commands at the send boundary. ROS 전송 경계에서 명령 수를 센다.
    실물 publisher 를 여기 끼우면 같은 경계를 지난다 — 검사와 실행이 같은 문을 쓴다."""

    def __init__(self) -> None:
        self.sent: list = []

    def send(self, waypoint: dict) -> None:
        self.sent.append(waypoint)

    @property
    def count(self) -> int:
        return len(self.sent)


def emit_trajectory(path, qs, gaps, dt_s, rep: dict, meta=None, publisher=None) -> dict:
    """Write and publish ONLY if preflight passed. 통과했을 때만 쓰고 보낸다.

    초판 결함 (황도경 지적 2026-09-20)
    ---------------------------------
    초판 main() 은 rep["ok"] 가 False 여도 write_trajectory() 를 **먼저** 불렀다.
    즉 검사는 실패했는데 실행 가능한 파일은 만들어졌다. 리포트가 실패를 적는 것과
    실행이 막히는 것은 다른 상태다. 이 함수가 그 둘을 하나로 묶는다.
    """
    if not rep.get("ok"):
        return {"written": None, "published": 0, "refused": list(rep.get("fail", []))}
    written = write_trajectory(path, qs, gaps, dt_s, meta)
    sent = 0
    if publisher is not None:
        for i, (q, g) in enumerate(zip(qs, gaps)):
            publisher.send({"t_s": round(i * dt_s, 6),
                            "q_rad": [float(v) for v in q],
                            "gap_m": float(g),
                            "gripper_tick": gap_to_tick(float(g))})
            sent += 1
    return {"written": written, "published": sent, "refused": []}


# ── 자체검사 ─────────────────────────────────────────────────────────────────
def _selftest() -> int:
    rows, fails = [], 0
    # 검사용 임계값. **실제 배포 설정이 아니다** — 배포본은 HW 승인 전까지 null 이라
    # 로딩 단계에서 거부된다. 그 거부 자체도 아래 [21-23] 에서 검사한다.
    SF = {"max_joint_speed_rad_s": 0.3, "max_step_tick": 300.0,
          "min_joint_margin_deg": 2.0, "wrist_roll_min_rad": -1.0472,
          "wrist_roll_max_rad": 1.1320, "min_tcp_z_table_mm": 15.0, "ik_tol_mm": 0.5}
    SF["max_step_rad"] = SF["max_step_tick"] * 2 * math.pi / TICKS_PER_TURN

    def chk(name, cond, note=""):
        nonlocal fails
        rows.append((name, bool(cond), note))
        if not cond:
            fails += 1

    rng = np.random.default_rng(0)
    R = K._rpy(0.3, -0.7, 1.1)[:3, :3]

    # [1-3] 회전 표현. 행/열을 구분 못 하면 이 검사는 검사가 아니다
    chk("1 rot6d 행 왕복", np.allclose(rot6d_to_matrix(matrix_to_rot6d(R, True), True), R))
    chk("2 행≠열 판별행", not np.allclose(rot6d_to_matrix(matrix_to_rot6d(R, True), False), R),
        "열로 읽으면 전치가 나와야 한다")
    chk("3 항등", np.allclose(rot6d_to_matrix(matrix_to_rot6d(np.eye(3))), np.eye(3)))

    # [4] 합성 순서
    A = np.eye(4); A[:3, :3] = K._rpy(0, 0, 0.5)[:3, :3]; A[:3, 3] = [0.01, 0, 0]
    Tc = np.eye(4); Tc[:3, :3] = R; Tc[:3, 3] = [0.2, 0.1, 0.1]
    chk("4 합성 순서 판별행", not np.allclose(Tc @ A, A @ Tc), "순서가 무의미하면 검사 실패")

    # [5-7] 액션 파싱
    ch = np.zeros((ACTION_HORIZON, ACTION_DIM)); ch[:, 3:9] = matrix_to_rot6d(np.eye(3))
    ch[:, 0] = 0.002; ch[:, 9] = 0.05
    p, g = unroll(np.eye(4), [ch, ch])
    chk("5 unroll 길이", len(p) == 2 * (EXEC_SLICE[1] - EXEC_SLICE[0]), f"{len(p)}/8")
    # [5b-5d] 앵커 규약 정답 아는 행 (D-AI-80). 길이만 보면 누적/앵커가 같은 출력이다
    _lo, _hi = EXEC_SLICE
    _n = _hi - _lo
    _exp = [0.002] * _n                                   # 앵커면 청크 안에서 전부 같다
    _got = [float(P[0, 3]) for P in p[:_n]]
    chk("5b 청크 안은 앵커다 (누적 아님)", np.allclose(_got, _exp, atol=1e-12),
        f"x={[round(v, 5) for v in _got]} · 기대 {_exp}")
    _accum = np.cumsum(_exp).tolist()                     # 누적이면 이렇게 나온다
    chk("5c 누적/앵커가 갈리는 입력이다 (판별행)", not np.allclose(_accum, _exp),
        f"누적이면 x={[round(v, 5) for v in _accum]}")
    chk("5d 청크 경계에서만 앵커가 넘어간다",
        abs(float(p[_n][0, 3]) - 0.004) < 1e-12, f"{float(p[_n][0, 3]):.6f} · 기대 0.004000")
    bad = ch[0].copy(); bad[9] = 0.5
    try:
        action_to_T(bad); chk("6 gap 범위 거부", False)
    except ValueError:
        chk("6 gap 범위 거부", True)
    try:
        action_to_T(np.zeros(9)); chk("7 차원 거부", False)
    except ValueError:
        chk("7 차원 거부", True)

    # [8] IK 왕복 — 순방향으로 만든 포즈라 반드시 풀려야 한다
    q_ref = np.array([0.1, -0.5, 0.6, -0.2, 0.0])
    T_ref = K.fk(q_ref)
    q_sol, r_mm, ok = K.ik(T_ref[:3, 3], approach=T_ref[:3, 2], finger=T_ref[:3, 0], q0=q_ref)
    chk("8 IK 왕복", ok and r_mm < SF["ik_tol_mm"], f"잔차 {r_mm:.4f}mm")

    # [9] jaw 축 — URDF 상수에서 직접
    T_ht = K._tf((0, -0.0780187530518, 0.0270000119209), (math.pi / 2, 0, 0))
    jaw = T_ht[:3, :3].T @ np.array([1.0, 0, 0])
    chk("9 jaw == TCP +x", np.allclose(jaw, JAW_AXIS_IN_TCP, atol=1e-12),
        f"보정 {math.degrees(math.acos(np.clip(jaw @ JAW_AXIS_IN_TCP, -1, 1))):.6f}deg")

    # [10-11] 속도 게이트 양방향. 하나만 있으면 판별력이 없다
    dt = 0.1
    step_ok = SF["max_joint_speed_rad_s"] * dt * 0.9
    step_no = SF["max_joint_speed_rad_s"] * dt * 1.1
    q_ok = np.cumsum(np.full((5, 5), step_ok), axis=0) * 0 + np.arange(5)[:, None] * step_ok
    q_no = np.tile(np.arange(5)[:, None] * step_no, (1, 5))
    chk("10 속도 통과", preflight(q_ok, [0.05] * 5, dt, None, SF)["checks"]["speed"]["violations"] == 0)
    chk("11 속도 거부 판별행", preflight(q_no, [0.05] * 5, dt, None, SF)["checks"]["speed"]["violations"] > 0)

    # [12] 빈 입력 — '없음' 과 '괜찮음' 이 같은 출력이면 안 된다
    empty = preflight(np.zeros((0, 5)), [], dt, None, SF)
    chk("12 빈 입력 불합격", (not empty["ok"]) and any("검사 0건" in f for f in empty["fail"]))

    # [13] 관절 한계 — 상한 밖
    hi = np.array(K.RAD_LIMITS).T[1]
    chk("13 관절한계 거부", preflight((hi * 1.01)[None, :], [0.05], dt, None, SF)["checks"]["joint_limit"]["violations"] == 1)
    try:
        preflight(np.zeros((3, 4)), [0.05] * 3, dt, None, SF); ok13b = False
    except ValueError:
        ok13b = True
    chk("13b 관절 수 불일치 거부 판별행", ok13b)

    # [14a/14b] min-z 양방향. 위반 자세를 실제로 하나 넣지 않으면 검사가 아니다
    q_low = np.array([-0.048678, 1.734001, -0.126454, -0.108621, -1.371851])  # z = -280.6mm (격자 탐색)
    z_low = K.table_from_base(K.fk(q_low)[:3, 3])[2]
    z_hi = K.table_from_base(K.fk(q_ref)[:3, 3])[2]
    chk("14a min-z 거부 판별행",
        z_low < SF["min_tcp_z_table_mm"] and preflight(q_low[None, :], [0.05], dt, None, SF)["checks"]["min_z"]["violations"] == 1,
        f"z {z_low:.1f}mm")
    chk("14b min-z 통과",
        z_hi >= SF["min_tcp_z_table_mm"] and preflight(q_ref[None, :], [0.05], dt, None, SF)["checks"]["min_z"]["violations"] == 0,
        f"z {z_hi:.1f}mm")

    # [15] 그리퍼 틱 — common.py 실측값
    chk("15 gap 0m -> 1763", gap_to_tick(0.0) == GRIPPER_CLOSED_TICK)
    chk("16 gap 0.09m -> 130", gap_to_tick(0.09) == GRIPPER_OPEN_TICK)

    # [17-19] A/B/C 대조군
    ch2 = np.zeros((ACTION_HORIZON, ACTION_DIM)); ch2[:, 3:9] = matrix_to_rot6d(np.eye(3))
    ch2[:, 0] = 0.001; ch2[:, 2] = 0.0005
    ch2[:, 9] = np.linspace(0.07, 0.04, ACTION_HORIZON)
    T0 = K.fk(q_ref)
    abc = make_abc([ch2, ch2, ch2], T0, q_ref, dt, SF, seed=7)
    qa, ga, _ = abc["A_policy"]; qb, gb, _ = abc["B_hold"]; qc, gc, _ = abc["C_shuffle"]
    chk("17 B 정지", np.allclose(np.diff(qb, axis=0), 0))
    chk("18 A≠B 판별행", not np.allclose(qa, qb))
    chk("19 A≠C 판별행", not np.allclose(ga, gc) or not np.allclose(qa, qc),
        "셔플이 원본과 같으면 대조군이 아니다")

    # [20] 파일 왕복
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = write_trajectory(Path(d) / "t.json", qa, ga, dt, {"test": True})
        doc = json.loads(p.read_text(encoding="utf-8"))
        chk("20 파일 왕복", doc["n_waypoints"] == len(qa) == len(doc["waypoints"])
            and doc["contractVersion"] == CONTRACT_VERSION)

    # [21-23] 안전 설정 fail-closed. "없음"과 "괜찮음"이 같은 출력이면 안 된다
    import tempfile as _tf
    with _tf.TemporaryDirectory() as d:
        dd = Path(d)
        try:
            load_safety(dd / "nope.json"); ok21 = False
        except SafetyUnset:
            ok21 = True
        chk("21 설정 파일 없음 -> 거부", ok21)

        full = {k: 1.0 for k in SAFETY_KEYS}
        full["wrist_roll_min_rad"], full["wrist_roll_max_rad"] = -1.0, 1.0
        for tag, mutate in (("22 null -> 거부", lambda c: c.update(min_joint_margin_deg=None)),
                            ("23 NaN -> 거부", lambda c: c.update(max_joint_speed_rad_s=float("nan")))):
            cfg = dict(full); mutate(cfg)
            f = dd / "s.json"; f.write_text(json.dumps(cfg), encoding="utf-8")
            try:
                load_safety(f); okx = False
            except SafetyUnset:
                okx = True
            chk(tag, okx)
        f = dd / "good.json"; f.write_text(json.dumps(full), encoding="utf-8")
        chk("23b 전부 채우면 통과 (판별행)", load_safety(f)["max_step_rad"] > 0)

    # [24-27] 불량 입력이 실제 전송 경계까지 못 가는지. 리포트 실패만으론 부족하다
    #   (황도경 지적 2026-09-20) — 검사 실패와 명령 차단은 다른 상태다.
    bad_roll = np.repeat(q_ref[None, :], 3, axis=0).copy()
    bad_roll[:, WRIST_ROLL_INDEX] = (337 - 2048) * 2 * math.pi / TICKS_PER_TURN   # -150.4도
    rep_r = preflight(bad_roll, [0.05] * 3, dt, None, SF)
    chk("24a wrist_roll 337틱 거부", rep_r["checks"]["wrist_roll_range"]["violations"] == 3
        and not rep_r["ok"], f"{math.degrees(bad_roll[0, WRIST_ROLL_INDEX]):.1f}deg")

    big = np.repeat(q_ref[None, :], 2, axis=0).copy()
    big[1, 1] += SF["max_step_rad"] * 1.5                                        # 300틱 초과
    rep_s = preflight(big, [0.05] * 2, dt, None, SF)
    chk("24b 300틱 초과 스텝 거부", rep_s["checks"]["max_step"]["violations"] >= 1 and not rep_s["ok"])

    with _tf.TemporaryDirectory() as d:
        for tag, qq, rr in (("25 wrist_roll 불량", bad_roll, rep_r), ("26 스텝 불량", big, rep_s)):
            pub = MockPublisher()
            out = emit_trajectory(Path(d) / "x.json", qq, [0.05] * len(qq), dt, rr, publisher=pub)
            chk(f"{tag} -> 전송 0회", pub.count == 0 and out["published"] == 0)
            chk(f"{tag} -> 궤적 미생성", out["written"] is None and not (Path(d) / "x.json").exists())

        # 판별행 — 정상 입력은 실제로 전송돼야 한다. 늘 0이면 검사가 아니다
        pub_ok = MockPublisher()
        rep_ok = preflight(qa, ga, dt, None, SF)
        out_ok = emit_trajectory(Path(d) / "y.json", qa, ga, dt, rep_ok, publisher=pub_ok)
        chk("27 정상 입력 전송 판별행",
            rep_ok["ok"] and pub_ok.count == len(qa) and out_ok["written"] is not None,
            f"전송 {pub_ok.count}/{len(qa)}")

    # [28] 관절 여유 — 모수와 최악 관절을 같이 낸다
    mrep = preflight(q_ref[None, :], [0.05], dt, None, SF)["checks"]["joint_margin"]
    chk("28 관절 여유 보고", mrep["checked"] == 1 and mrep["trajectory_min_deg"] is not None
        and mrep["tightest_joint_index"] is not None, f"최소 {mrep['trajectory_min_deg']}도")

    for name, ok, note in rows:
        print(f"  {'OK ' if ok else 'FAIL'}  {name}" + (f"   {note}" if note else ""))
    print(f"\n자체검사 {len(rows) - fails}/{len(rows)} 통과")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--actions", help="(N,8,10) 액션 청크 .npy")
    ap.add_argument("--q0", help="시작 관절각 5개 rad, 콤마 구분")
    ap.add_argument("--dt", type=float, default=1.0 / NOMINAL_RATE_HZ,
                    help="경유점 간격 s. 공칭 0.1 — 실제 간격이 다르면 반드시 넘겨라")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--safety", default=None,
                    help=f"안전 임계값 JSON. 기본 {SAFETY_PATH}. 누락·null·NaN 이면 실행 거부")
    ap.add_argument("--publish", action="store_true",
                    help="전송 경계를 실제로 통과시킨다 (지금은 MockPublisher 로 센다)")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not (a.actions and a.q0 and a.out_dir):
        ap.error("--actions, --q0, --out-dir 이 모두 필요하다")

    try:
        safety = load_safety(a.safety)
    except SafetyUnset as exc:
        print(f"!! 안전 설정 미확정 — 한 점도 내보내지 않는다.\n{exc}")
        return 2
    print(f"안전 설정 {a.safety or SAFETY_PATH} 로드: "
          + " · ".join(f"{k}={safety[k]}" for k in SAFETY_KEYS))

    chunks = np.load(a.actions)
    if chunks.ndim == 2:
        chunks = chunks[None, ...]
    q0 = np.array([float(x) for x in a.q0.split(",")], float)
    print(f"입력 청크 {len(chunks)} · 경유점 예정 {len(chunks) * (EXEC_SLICE[1] - EXEC_SLICE[0])} · dt {a.dt}s")

    abc = make_abc(chunks, K.fk(q0), q0, a.dt, safety, seed=a.seed)
    worst = 0
    for name, (qs, gaps, rep) in abc.items():
        pub = MockPublisher() if a.publish else None
        out = emit_trajectory(Path(a.out_dir) / f"{name}.json", qs, gaps, a.dt, rep,
                              {"arm_variant": "so101_ver1", "source_actions": a.actions,
                               "seed": a.seed, "preflight": rep}, publisher=pub)
        p = out["written"] or "(생성 안 함 — 프리플라이트 불합격)"
        print(f"\n[{name}] -> {p}   전송 {out['published']}건")
        for k, c in rep["checks"].items():
            extra = ""
            if k == "speed" and c["violations"]:
                extra = f"   -> dt {c['required_dt_s']}s 이상 필요 ({c['slowdown_x']}배 감속)"
            print(f"   {k:14s} 위반 {c['violations']}/{c['checked']}{extra}")
        if not rep["ok"]:
            print("   !! " + " · ".join(rep["fail"]))
            worst = max(worst, 1 if name == "A_policy" else 0)
    print("\nA 가 불합격이면 실물에 넣지 마라.")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
