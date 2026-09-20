"""Continuous-IK feasibility of real UMI demos under simulation-only start alignment.
simulation-only episode-start alignment 기준으로 실 UMI 시연의 연속 IK 실행가능성을 잰다.

무엇을 재나
-----------
v10(`umi_relative_chunk/0.2.0-provisional`)은 절대 pose 를 담지 않는다.
`action[i,0]` 이 현재 pinch pose 기준 +0.1초 pose 의 상대량이므로, 이를 연쇄하면
에피소드 전체의 상대 궤적이 복원된다. 그 궤적의 시작을 시뮬 홈 EEF pose 에 박는다.

    T_base[0] = T_base_home
    T_base[i] = T_base[i-1] @ A_i0          (A_i0 = action[i-1,0] 의 변환)

⚠️ 이것은 **실물 베이스 캘리브레이션이 아니다.** simulation-only episode-start
alignment 이고, 리포트의 모든 행에 그렇게 표기된다.

계측기 자체 검증
----------------
- `--selftest`: rot6d 왕복, 연쇄 복원, 알려진 궤적에 대한 IK 를 먼저 확인한다
- 복원 교차검증: `action[i,0]` 연쇄로 만든 절대 궤적과 `action[i,k]`(k>0)가
  가리키는 pose 를 대조한다. 어긋나면 **복원이 틀린 것이므로 수용률을 내지 않는다**
- 관절 속도 한계는 **발명하지 않는다.** 시뮬 전문가 시연에서 실측한 최대
  관절 속도를 예산으로 쓰고(`--reference-demos`), 실 궤적의 요구치를 그 배수로 낸다

Usage
-----
  python check_real_traj_ik.py --selftest
  python check_real_traj_ik.py --dataset ~/S15P21A103_umi/AI/datasets/umi_real_relative_20260911_v10 \
      --task ~/handoff/configs/can_side.yaml --reference-demos ~/handoff/outputs/demos_6000 \
      --out ~/handoff/outputs/ik_feasibility.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

RATE_HZ = 10.0
DT = 1.0 / RATE_HZ
ACTION_COLUMNS = ["x_m", "y_m", "z_m", "r0x", "r0y", "r0z", "r1x", "r1y", "r1z", "gap_m"]


# ── 회전 표현 ────────────────────────────────────────────────────────────

def rot6d_to_matrix(r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """Gram-Schmidt a 6D rotation representation into a 3x3 matrix.
    6D 회전 표현을 그람-슈미트로 3x3 행렬로 만든다."""
    b0 = np.asarray(r0, dtype=np.float64)
    n0 = np.linalg.norm(b0)
    if n0 < 1e-9:
        raise ValueError(f"r0 의 크기가 0에 가깝다: {n0:.3e}")
    b0 = b0 / n0
    a1 = np.asarray(r1, dtype=np.float64)
    b1 = a1 - np.dot(b0, a1) * b0
    n1 = np.linalg.norm(b1)
    if n1 < 1e-9:
        raise ValueError(f"r0 와 r1 이 평행하다 (잔차 {n1:.3e})")
    b1 = b1 / n1
    b2 = np.cross(b0, b1)
    # ⚠️ r0,r1 은 회전행렬의 **행**이다. 2026-09-16 규약 스윕에서 8개 조합 중
    #    (행 기준 · T_next = T_cur @ A) 만 교차검증 오차 0.0000mm 였다.
    #    열로 읽으면 중앙오차 4mm 가 남는다 — 조용히 틀린 채로 돈다.
    return np.column_stack([b0, b1, b2]).T


def matrix_to_rot6d(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of rot6d_to_matrix (first two rows). 첫 두 행을 돌려준다."""
    return R[0, :].copy(), R[1, :].copy()


def row_to_transform(row: np.ndarray) -> np.ndarray:
    """One 10-column action row -> 4x4 homogeneous transform (gap ignored).
    10열 행 하나를 4x4 동차변환으로. gap 은 빼고 본다."""
    T = np.eye(4)
    T[:3, :3] = rot6d_to_matrix(row[3:6], row[6:9])
    T[:3, 3] = row[0:3]
    return T


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees. 두 회전의 측지 각도[deg]."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


# ── 궤적 복원 ────────────────────────────────────────────────────────────

def reconstruct_relative_chain(action: np.ndarray) -> np.ndarray:
    """Chain action[i,0] into absolute-from-start transforms.
    action[i,0] 을 연쇄해 시작 기준 절대 변환열을 만든다.

    반환: (N+1, 4, 4). index 0 은 항등(시작 pose)."""
    n = action.shape[0]
    out = np.zeros((n + 1, 4, 4))
    out[0] = np.eye(4)
    for i in range(n):
        out[i + 1] = out[i] @ row_to_transform(action[i, 0])
    return out


def chain_consistency(action: np.ndarray, chain: np.ndarray, horizon: int) -> dict:
    """Cross-check the chain against multi-step action targets.
    연쇄 복원을 다단계 action 타깃과 대조한다. 어긋나면 복원이 틀린 것이다."""
    pos_err, rot_err, n_cmp = [], [], 0
    N = action.shape[0]
    for i in range(N):
        for k in range(1, horizon):
            j = i + k + 1              # chain index of the target pose
            if j >= chain.shape[0]:
                continue
            predicted = chain[i] @ row_to_transform(action[i, k])
            actual = chain[j]
            pos_err.append(float(np.linalg.norm(predicted[:3, 3] - actual[:3, 3])))
            rot_err.append(geodesic_deg(predicted[:3, :3], actual[:3, :3]))
            n_cmp += 1
    if n_cmp == 0:
        return {"comparisons": 0, "position_max_mm": None, "rotation_max_deg": None}
    return {
        "comparisons": n_cmp,
        "position_median_mm": float(np.median(pos_err) * 1000),
        "position_max_mm": float(np.max(pos_err) * 1000),
        "rotation_median_deg": float(np.median(rot_err)),
        "rotation_max_deg": float(np.max(rot_err)),
    }


# ── 속도 예산 (시뮬 전문가에서 실측) ──────────────────────────────────────

def measure_reference_limits(demos_dir: Path, max_episodes: int = 20) -> dict:
    """Measure joint speed/accel budget from the scripted expert's own demos.
    시뮬 전문가 시연에서 관절 속도·가속도 예산을 실측한다. 값을 발명하지 않는다."""
    eps = sorted(p for p in demos_dir.glob("episode_*") if p.is_dir())[:max_episodes]
    if not eps:
        raise SystemExit(f"!! 참조 시연이 없다: {demos_dir}")
    vmax, amax, used = [], [], 0
    for e in eps:
        f = e / "recording.npz"
        if not f.exists():
            continue
        z = np.load(f)
        if "joint_pos" not in z or "timestamp" not in z:
            continue
        q = np.asarray(z["joint_pos"], dtype=np.float64)
        t = np.asarray(z["timestamp"], dtype=np.float64)
        if len(t) < 3:
            continue
        dt = np.diff(t)[:, None]
        if np.any(dt <= 0):
            continue
        v = np.diff(q, axis=0) / dt
        a = np.diff(v, axis=0) / dt[1:]
        vmax.append(np.abs(v).max(axis=0))
        amax.append(np.abs(a).max(axis=0))
        used += 1
    if used == 0:
        raise SystemExit(f"!! 참조 시연 {len(eps)}편 중 쓸 수 있는 게 0편이다 "
                         f"(joint_pos/timestamp 필드 확인)")
    return {
        "episodes_used": used,
        "episodes_found": len(eps),
        "joint_speed_rad_s": np.max(np.stack(vmax), axis=0).tolist(),
        "joint_accel_rad_s2": np.max(np.stack(amax), axis=0).tolist(),
    }


# ── 에피소드 검사 ────────────────────────────────────────────────────────

def grasp_window(z, chain_len: int, pre: int, post: int) -> tuple[int, int, int]:
    """Rows around the closure moment — what the task actually requires.
    파지(그리퍼 닫힘) 순간 주변 구간. 태스크가 실제로 요구하는 곳이다.

    ⚠️ 2026-09-17 정정 — 이 도구의 첫 판은 **시연 전체 궤적**의 재현 가능성을
    쟀다. 그건 "사람 손을 그대로 따라 하는가" 를 묻는 것이고, 이 제품의 기조가
    아니다. 사람이 요란하게 움직여도 로봇은 물체만 같게 집으면 된다.
    그래서 접근 중 휘저은 구간까지 실패로 세면 폐기율이 크게 부풀려진다.

    닫힘 순간은 gap_m(절대 미터)의 최소점으로 잡는다. proprio 의 마지막 열이다.
    """
    pr = np.asarray(z["proprio"], dtype=np.float64)
    gap = pr[:, -1, 9]                      # 각 행의 현재 gap[m]
    if gap.size == 0:
        raise ValueError("proprio 가 비었다")
    c = int(np.argmin(gap))
    lo = max(0, c - pre)
    hi = min(chain_len - 1, c + post)
    return lo, hi, c


def step_durations(z, n_steps: int) -> np.ndarray:
    """Real seconds between consecutive waypoints, from recorded timestamps.
    웨이포인트 사이 실제 초. 기록된 타임스탬프에서 읽는다.

    ⚠️ `source_row` 간격이 균일하지 않다(3 과 2 가 섞인다). 0.1초 고정으로
    계산하면 속도가 최대 1.5배 어긋난다. 고정값을 쓰지 않는다."""
    ot = np.asarray(z["observation_timestamp"], dtype=np.float64)
    cur = ot[:, -1]                       # 각 행의 현재 관측 시각
    if len(cur) < 2:
        return np.full(max(n_steps, 1), np.nan)
    d = np.diff(cur)
    if np.any(d <= 0):
        raise ValueError("observation_timestamp 가 단조증가가 아니다")
    return np.r_[d, d[-1]]                # 마지막은 직전 간격으로 채운다


def rotation_error_split(R_target: np.ndarray, R_actual: np.ndarray) -> tuple[float, float]:
    """Split the orientation error into the free axis and the rest [deg].
    자세 오차를 **자유 축 성분**과 **나머지**로 가른다.

    왜 나누나 / Why split
    ---------------------
    평행 그리퍼 측면 파지는 **접근축(TCP z) 둘레 회전이 자유**다 (프로젝트 기조).
    그 축 둘레로 틀어진 것은 파지에 해가 없는데, 측지각 하나로 재면 해로운 오차와
    **같은 숫자**가 된다. 그러면 "5도 초과 거부" 가 무해한 것까지 거부한다.

    ⚠️ 이 함수는 **보고만 한다.** 거부 규칙은 바꾸지 않는다 — 결과를 보고 게이트를
    옮기는 것이 되기 때문이다. 바꾸려면 사전등록이 먼저다.

    Returns (접근축 둘레 성분, 나머지 성분) [deg].
    """
    Rerr = np.asarray(R_target).T @ np.asarray(R_actual)
    # 회전벡터로 펴서 목표 프레임의 z(접근축)에 투영한다.
    c = (float(np.trace(Rerr)) - 1.0) / 2.0
    ang = float(np.arccos(max(-1.0, min(1.0, c))))
    if ang < 1e-12:
        return 0.0, 0.0
    axis = np.array([Rerr[2, 1] - Rerr[1, 2],
                     Rerr[0, 2] - Rerr[2, 0],
                     Rerr[1, 0] - Rerr[0, 1]]) / (2.0 * np.sin(ang))
    v = axis * ang                                  # 목표 프레임에서 본 회전벡터
    about = abs(float(v[2]))                        # z = 접근축
    perp = float(np.linalg.norm(v[:2]))
    return float(np.degrees(about)), float(np.degrees(perp))


def check_episode(env, chain: np.ndarray, T_base_home: np.ndarray, limits: dict,
                  horizon: int, ik_tol: float, dts: np.ndarray,
                  span: tuple[int, int] | None = None) -> dict:
    """Continuous IK over one episode: previous solution seeds the next waypoint.
    한 에피소드의 연속 IK. 이전 해가 다음 웨이포인트의 시드가 된다."""
    import mujoco

    vbud = np.asarray(limits["joint_speed_rad_s"], dtype=np.float64)
    abud = np.asarray(limits["joint_accel_rad_s2"], dtype=np.float64)
    # ⚠️ 2026-09-19 수정 — 이 두 줄이 아래 `lo, hi = span ...` 에 **덮어써지고 있었다.**
    #    그 결과 관절 한계 검사가 `q < 구간인덱스` 를 비교했고, 관절각(rad)은 거의 항상
    #    그 정수보다 작아서 **전부 joint_limit 으로 거부**됐다.
    #    0917 MEASURE §7 에 결함 #11 로 적혀 있었으나 **수정이 저장소에 들어오지 않았다.**
    #    이름을 분리한다. 같은 이름을 두 뜻으로 쓰지 않는다.
    q_lo, q_hi = env.limits[:, 0], env.limits[:, 1]

    qs, reasons, residuals, splits = [], [], [], []
    margins: list = []                        # [joint_margin_patch]
    seed = env.home_q.copy()
    first_fail = None

    lo, hi = span if span else (0, chain.shape[0] - 1)
    for idx in range(lo, hi + 1):
        T = T_base_home @ chain[idx]
        pos, R = T[:3, 3], T[:3, :3]
        why = None
        try:
            q = env.ik(pos, R, seed=seed, strict=False)
        except Exception as exc:                      # noqa: BLE001 — 사유를 남긴다
            q = None
            why = f"ik_exception:{type(exc).__name__}"
        if q is not None:
            env.ik_data.qpos[env.qids] = q
            mujoco.mj_forward(env.model, env.ik_data)
            Tc = env.tcp(env.ik_data)
            e_pos = float(np.linalg.norm(Tc[:3, 3] - pos))
            e_rot = geodesic_deg(R, Tc[:3, :3])
            e_free, e_perp = rotation_error_split(R, Tc[:3, :3])
            residuals.append((e_pos, e_rot))
            splits.append((e_free, e_perp))
            if np.any(q < q_lo - 1e-6) or np.any(q > q_hi + 1e-6):
                why = "joint_limit"
            elif e_pos > ik_tol:
                why = f"position_residual>{ik_tol}"
            elif e_rot > 5.0:
                why = "rotation_residual>5deg"
        else:
            residuals.append((float("nan"), float("nan")))
            splits.append((float("nan"), float("nan")))
        if why is None:
            qs.append(q)
            seed = q
            # [joint_margin_patch] 한계까지 남은 여유를 관절별로 적재한다.
            #    통과 여부만으로는 아슬아슬한 편과 여유 있는 편이 구분되지 않는다.
            margins.append(np.minimum(np.asarray(q) - q_lo, q_hi - np.asarray(q)))
        else:
            qs.append(None)
            if first_fail is None:
                first_fail = {"index": idx, "reason": why}
        reasons.append(why)

    ok_way = [r is None for r in reasons]

    # 속도·가속도: 연속으로 성공한 구간에서만 본다
    vio_v = vio_a = 0
    need_scale = 1.0
    for i in range(1, len(qs)):
        if qs[i] is None or qs[i - 1] is None:
            continue
        dt = float(dts[i - 1]) if i - 1 < len(dts) else float("nan")
        if not np.isfinite(dt) or dt <= 0:
            continue
        v = np.abs(np.asarray(qs[i]) - np.asarray(qs[i - 1])) / dt
        ratio = float(np.max(v / np.maximum(vbud, 1e-9)))
        if ratio > 1.0:
            vio_v += 1
            need_scale = max(need_scale, ratio)
        if i >= 2 and qs[i - 2] is not None:
            a = np.abs(np.asarray(qs[i]) - 2 * np.asarray(qs[i - 1]) + np.asarray(qs[i - 2])) / (dt * dt)
            ar = float(np.max(a / np.maximum(abud, 1e-9)))
            if ar > 1.0:
                vio_a += 1
                need_scale = max(need_scale, math.sqrt(ar))

    n_chunks = max(0, len(ok_way) - horizon)
    ok_chunks = sum(1 for i in range(n_chunks) if all(ok_way[i:i + horizon]))
    fin = [r for r in residuals if not math.isnan(r[0])]
    _free = [a for a, _ in splits if not math.isnan(a)]
    _perp = [b for _, b in splits if not math.isnan(b)]

    too_short = len(ok_way) < horizon + 1
    return {
        "span": [lo, hi],
        "too_short_for_chunk": bool(too_short),
        "waypoints": len(ok_way),
        "waypoints_ok": int(sum(ok_way)),
        "chunks": n_chunks,
        "chunks_ok": ok_chunks,
        "episode_ok": bool(all(ok_way)) and vio_v == 0 and vio_a == 0,
        "first_failure": first_fail,
        "velocity_violations": vio_v,
        "acceleration_violations": vio_a,
        "required_uniform_time_scale": round(need_scale, 4),
        "position_residual_max_mm": round(max((r[0] for r in fin), default=float("nan")) * 1000, 3) if fin else None,
        "rotation_residual_max_deg": round(max((r[1] for r in fin), default=float("nan")), 3) if fin else None,
        # 보고용 분해 (거부 규칙에는 안 쓴다)
        # 빈 목록에 max 를 걸지 않는다. 비었으면 None 이고, 그건 "0도" 와 다른 상태다.
        "rot_free_axis_max_deg": (round(max(_free), 3) if _free else None),
        "rot_perp_max_deg": (round(max(_perp), 3) if _perp else None),
        "rot_split_n": len(_free),
        "rot_split_note": "접근축 둘레 성분은 평행 그리퍼 측면파지에서 자유 축이다. "
                          "거부 판정에는 쓰지 않았다",
        "reasons": reasons,
        # [joint_margin_patch] 여유 계측. 모수(scored/총)를 항상 같이 낸다.
        **_margin_block(margins, len(ok_way)),
    }


def _margin_block(margins: list, n_waypoints: int) -> dict:
    """Summarize per-joint distance to the nearest joint limit, in degrees.
    관절별 한계까지 남은 여유를 도 단위로 요약한다.

    IK 가 한 번도 안 풀린 편은 여유를 잴 수 없다. 값은 None 으로 두되 사유를 같이 실어서
    '여유 없음' 과 '여유 못 쟀음' 이 같은 출력이 되지 않게 한다.
    """
    scored = len(margins)
    if scored == 0:
        return {
            "margin_scored_waypoints": 0,
            "margin_total_waypoints": int(n_waypoints),
            "joint_margin_min_deg": None,
            "joint_margin_min_overall_deg": None,
            "tightest_joint_index": None,
            "margin_unmeasured_reason": "no_successful_ik — 여유를 못 쟀다. 여유가 넉넉한 것이 아니다",
        }
    M = np.degrees(np.stack(margins))
    per_joint = M.min(axis=0)
    j = int(np.argmin(per_joint))
    return {
        "margin_scored_waypoints": int(scored),
        "margin_total_waypoints": int(n_waypoints),
        "joint_margin_min_deg": [round(float(v), 3) for v in per_joint],
        "joint_margin_min_overall_deg": round(float(per_joint[j]), 3),
        "tightest_joint_index": j,
        "margin_unmeasured_reason": None,
    }


# ── 자체 검증 ────────────────────────────────────────────────────────────

def selftest() -> int:
    """Verify the instrument before trusting it. 쓰기 전에 계측기를 검증한다."""
    rng = np.random.default_rng(0)
    bad = 0

    # 1) rot6d 왕복
    errs = []
    for _ in range(2000):
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        r0, r1 = matrix_to_rot6d(Q)
        errs.append(np.abs(rot6d_to_matrix(r0, r1) - Q).max())
    print(f"[1] rot6d 왕복 2000회 최대오차 {max(errs):.3e}", end="  ")
    print("OK" if max(errs) < 1e-9 else "!! 실패"); bad += max(errs) >= 1e-9

    # 2) 알려진 궤적을 연쇄로 복원
    N, H = 12, 8
    step = np.eye(4)
    step[:3, 3] = [0.01, -0.005, 0.002]
    ang = math.radians(2.0)
    step[:3, :3] = np.array([[math.cos(ang), -math.sin(ang), 0],
                             [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
    truth = [np.eye(4)]
    for _ in range(N + H):
        truth.append(truth[-1] @ step)
    action = np.zeros((N, H, 10))
    for i in range(N):
        for k in range(H):
            rel = np.linalg.inv(truth[i]) @ truth[i + k + 1]
            r0, r1 = matrix_to_rot6d(rel[:3, :3])
            action[i, k] = np.r_[rel[:3, 3], r0, r1, 0.05]
    chain = reconstruct_relative_chain(action)
    e = max(np.abs(chain[i] - truth[i]).max() for i in range(N + 1))
    print(f"[2] 연쇄 복원 최대오차 {e:.3e}", end="  ")
    print("OK" if e < 1e-9 else "!! 실패"); bad += e >= 1e-9

    # 3) 교차검증이 정상 데이터에서 0 을 내는가
    cc = chain_consistency(action, chain, H)
    print(f"[3] 교차검증 비교 {cc['comparisons']}건 위치최대 {cc['position_max_mm']:.6f}mm "
          f"회전최대 {cc['rotation_max_deg']:.6f}deg", end="  ")
    ok3 = cc["comparisons"] > 0 and cc["position_max_mm"] < 1e-6
    print("OK" if ok3 else "!! 실패"); bad += not ok3

    # 4) 교차검증이 망가진 데이터를 **잡는가** (검사가 검사 노릇을 하는지)
    broken = action.copy()
    broken[3, 4, 0] += 0.05
    cb = chain_consistency(broken, chain, H)
    caught = cb["position_max_mm"] > 40.0
    print(f"[4] 고의 손상 감지 위치최대 {cb['position_max_mm']:.3f}mm", end="  ")
    print("OK" if caught else "!! 실패 — 망가진 데이터를 못 잡는다"); bad += not caught

    # [N] 자세 오차 분해 — 정답 아는 행 + 판별력

    def _rot(ax, deg):

        a = np.deg2rad(deg)

        v = np.asarray(ax, float) / np.linalg.norm(ax)

        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


    _f, _p = rotation_error_split(np.eye(3), _rot([0, 0, 1], 7.0))

    _ok1 = abs(_f - 7.0) < 1e-6 and _p < 1e-6

    _f2, _p2 = rotation_error_split(np.eye(3), _rot([1, 0, 0], 7.0))

    _ok2 = _f2 < 1e-6 and abs(_p2 - 7.0) < 1e-6

    _f3, _p3 = rotation_error_split(np.eye(3), np.eye(3))

    print(f"[{'OK' if (_ok1 and _ok2 and _f3 == 0) else '!!'}] 자세오차 분해 "

          f"자유축7도->({_f:.3f},{_p:.3f}) 수직7도->({_f2:.3f},{_p2:.3f}) "

          f"동일->({_f3:.3f},{_p3:.3f})")

    if not (_ok1 and _ok2):

        raise SystemExit("!! 자세오차 분해가 축을 못 가른다")


    # [N] 이름 충돌 재발 방지 (2026-09-19). 관절 한계와 구간 인덱스가 같은 이름을

    #     쓰면 검사가 조용히 무력화된다. 소스에서 직접 확인한다.

    src = Path(__file__).read_text(encoding="utf-8")

    body = src[src.index("def check_episode("):src.index("# \u2500\u2500 \uc790\uccb4 \uac80\uc99d")]

    # ⚠️ 2026-09-20 정정 — 초판은 여기서 `bad` 에 **재대입**했다. 그 순간 위 [1]~[4]
    #    의 실패 카운트가 통째로 사라져, rot6d 를 열로 되돌려도 "자체검증 통과 · EXIT 0"
    #    이 나왔다(실증). 247행 주석이 경고한 `lo, hi` 이름 충돌과 **같은 종류의 실수**가
    #    같은 파일 안에서 재발했다. 이름을 분리한다.
    name_clash = ("lo, hi = env.limits" in body) and ("lo, hi = span" in body)
    bad += 1 if name_clash else 0

    print(f"[{'!!' if name_clash else 'OK'}] 관절한계·구간인덱스 이름 분리 "

          f"(q_lo/q_hi {'있음' if 'q_lo' in body else '없음'})")

    if name_clash:

        raise SystemExit("!! 관절 한계 변수가 구간 인덱스에 덮어써진다. 수치를 내지 않는다")


    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


# ── 본체 ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", help="v10 디렉터리")
    ap.add_argument("--task", help="시뮬 task yaml (홈 pose 를 여기서 얻는다)")
    ap.add_argument("--reference-demos", help="속도 예산을 실측할 시뮬 시연 디렉터리")
    ap.add_argument("--out", default="ik_feasibility.json")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--ik-tol", type=float, default=0.008,
                    help="위치 잔차 허용치[m]. env.ik 의 strict 임계와 같은 값")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N편만 (0=전부)")
    ap.add_argument("--segment", choices=["full", "grasp"], default="full",
                    help="full=시연 전체 궤적 재현 / grasp=파지 순간 주변만. "
                         "제품 기조는 결과 모방이므로 grasp 가 기본 판정 대상이다")
    ap.add_argument("--pre", type=int, default=2, help="--segment grasp: 닫힘 전 몇 행부터")
    ap.add_argument("--post", type=int, default=6, help="--segment grasp: 닫힘 후 몇 행까지")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    for need in ("dataset", "task", "reference_demos"):
        if not getattr(a, need):
            raise SystemExit(f"!! --{need.replace('_','-')} 가 필요하다 (또는 --selftest)")

    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 수용률을 내지 않는다")
    print()

    d = Path(a.dataset).expanduser()
    meta = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
    eps = meta["episodes"][: a.limit] if a.limit else meta["episodes"]
    print(f"데이터셋 {meta['schema']} · 전체 {meta['n_episodes']}편 · 검사 {len(eps)}편")
    if abs(float(meta.get("rate_hz", 0)) - RATE_HZ) > 1e-6:
        raise SystemExit(f"!! rate_hz 가 {meta.get('rate_hz')} 다. 이 도구는 {RATE_HZ} 전용이다")
    if int(meta.get("action_horizon", 0)) != a.horizon:
        raise SystemExit(f"!! action_horizon 이 {meta.get('action_horizon')} 인데 "
                         f"--horizon {a.horizon} 로 돌리려 한다")

    limits = measure_reference_limits(Path(a.reference_demos).expanduser())
    print(f"속도 예산 실측: {limits['episodes_used']}/{limits['episodes_found']}편 사용")
    print("  관절 최대속도[rad/s]  ", np.round(limits["joint_speed_rad_s"], 3))
    print("  관절 최대가속도[rad/s2]", np.round(limits["joint_accel_rad_s2"], 1))

    from simulation.env import PickEnv
    env = PickEnv(task_path=str(Path(a.task).expanduser()))
    env.reset(0)
    T_base_home = env.tcp().copy()
    print("시뮬 홈 EEF 위치", np.round(T_base_home[:3, 3], 4), "\n")

    results, skipped = [], []
    for name in eps:
        f = d / f"{name}.npz"
        if not f.exists():
            skipped.append({"episode": name, "reason": "npz 없음"})
            continue
        z = np.load(f)
        action = np.asarray(z["action"], dtype=np.float64)
        if action.shape[1] != a.horizon or action.shape[2] != len(ACTION_COLUMNS):
            skipped.append({"episode": name, "reason": f"action shape {action.shape}"})
            continue
        chain = reconstruct_relative_chain(action)
        cc = chain_consistency(action, chain, a.horizon)
        dts = step_durations(z, chain.shape[0])
        span = None
        closure = None
        if a.segment == "grasp":
            lo, hi, closure = grasp_window(z, chain.shape[0], a.pre, a.post)
            if hi - lo < 2:
                skipped.append({"episode": name, "reason": f"파지 구간이 {hi-lo+1}행뿐"})
                continue
            span = (lo, hi)
        r = check_episode(env, chain, T_base_home, limits, a.horizon, a.ik_tol, dts, span)
        r["closure_row"] = closure
        r["median_step_seconds"] = round(float(np.nanmedian(dts)), 4) if len(dts) else None
        r["rows"] = int(action.shape[0])
        r["episode"] = name
        r["chain_consistency"] = cc
        r["chain_trusted"] = bool(cc["comparisons"] > 0 and cc["position_max_mm"] < 1.0)
        results.append(r)
        flag = "" if r["chain_trusted"] else "  !! 복원 불일치"
        print(f"{name}  waypoint {r['waypoints_ok']}/{r['waypoints']}  "
              f"chunk {r['chunks_ok']}/{r['chunks']}  "
              f"episode {'O' if r['episode_ok'] else 'X'}  "
              f"timescale {r['required_uniform_time_scale']}{flag}", flush=True)
    env.close()

    if not results:
        raise SystemExit("!! 검사된 에피소드가 0편이다")

    untrusted = [r["episode"] for r in results if not r["chain_trusted"]]
    short = [r["episode"] for r in results if r["too_short_for_chunk"]]
    usable = [r for r in results if not r["too_short_for_chunk"] and r["chain_trusted"]]
    wp = sum(r["waypoints_ok"] for r in results), sum(r["waypoints"] for r in results)
    ch = sum(r["chunks_ok"] for r in results), sum(r["chunks"] for r in results)
    ep_ok = sum(1 for r in results if r["episode_ok"])

    # [joint_margin_patch] 편 단위 여유 분포. 분모를 항상 같이 찍는다.
    def margin_summary(results: list) -> None:
        scored = [r for r in results if r.get("joint_margin_min_overall_deg") is not None]
        unscored = len(results) - len(scored)
        print(f"\n관절 여유 — 잰 편 {len(scored)}/{len(results)} (못 잰 편 {unscored}: IK 전무)")
        if not scored:
            print("  ⚠️ 한 편도 못 쟀다. 여유가 넉넉한 것이 아니라 계측이 안 된 것이다")
            return
        vals = sorted(r["joint_margin_min_overall_deg"] for r in scored)
        for thr in (1.0, 5.0, 10.0):
            print(f"  여유 {thr:4.1f}도 미만  {sum(1 for v in vals if v < thr)}/{len(scored)}편")
        print(f"  최소 {vals[0]:.2f}도 · 중앙 {vals[len(vals) // 2]:.2f}도 · 최대 {vals[-1]:.2f}도")
        from collections import Counter
        c = Counter(r["tightest_joint_index"] for r in scored)
        print("  가장 빡빡한 관절: " + " · ".join(
            f"j{k} {v}/{len(scored)}편" for k, v in sorted(c.items())))

    def rate(x, n):
        return round(100.0 * x / n, 1) if n else None

    reason_counts: dict[str, int] = {}
    for r in results:
        for why in r["reasons"]:
            if why:
                reason_counts[why] = reason_counts.get(why, 0) + 1

    summary = {
        "alignment": "simulation-only episode-start alignment "
                     "(NOT a physical robot-base calibration)",
        "segment": a.segment,
        "segment_note": ("full = 시연 전체 궤적 재현 (사람 손 모방). "
                         "grasp = 파지 순간 주변만 (결과 모방, 제품 기조)"),
        "grasp_window": {"pre": a.pre, "post": a.post} if a.segment == "grasp" else None,
        "dataset": str(d),
        "dataset_schema": meta["schema"],
        "task": str(a.task),
        "rate_hz": RATE_HZ,
        "action_horizon": a.horizon,
        "ik_tolerance_m": a.ik_tol,
        "reference_limits": limits,
        "episodes_checked": len(results),
        "episodes_skipped": skipped,
        "chain_untrusted_episodes": untrusted,
        "too_short_episodes": short,
        "usable_episodes": len(usable),
        "rotation_convention": "r0,r1 = first two ROWS of R (2026-09-16 규약 스윕으로 확정)",
        "composition": "T_next = T_cur @ A_relative",
        "episode_acceptance_over_usable": {
            "ok": sum(1 for r in usable if r["episode_ok"]),
            "total": len(usable),
            "percent": rate(sum(1 for r in usable if r["episode_ok"]), len(usable)),
        },
        "waypoint_acceptance": {"ok": wp[0], "total": wp[1], "percent": rate(*wp)},
        "chunk_acceptance": {"ok": ch[0], "total": ch[1], "percent": rate(*ch)},
        "episode_acceptance": {"ok": ep_ok, "total": len(results),
                               "percent": rate(ep_ok, len(results))},
        "rejection_reasons": reason_counts,
        "required_time_scale_max": max(r["required_uniform_time_scale"] for r in results),
        "episodes": results,
    }
    Path(a.out).expanduser().write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                        encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"구간: {a.segment}" + (f" (닫힘 -{a.pre} ~ +{a.post}행)" if a.segment == "grasp" else " (시연 전체)"))
    print(f"waypoint  {wp[0]}/{wp[1]} = {rate(*wp)}%")
    print(f"chunk     {ch[0]}/{ch[1]} = {rate(*ch)}%")
    print(f"episode   {ep_ok}/{len(results)} = {rate(ep_ok, len(results))}%  (전체)")
    uo = sum(1 for r in usable if r["episode_ok"])
    print(f"episode   {uo}/{len(usable)} = {rate(uo, len(usable))}%  "
          f"(청크 가능 + 복원 신뢰 편만)   ← 게이트 판정 대상")
    if short:
        print(f"!! 청크를 만들 수 없는 짧은 편 {len(short)}편 (행 < horizon+1): {short[:5]}")
    print(f"필요 시간확대율 최대 {summary['required_time_scale_max']}")
    if untrusted:
        print(f"!! 복원 불일치 {len(untrusted)}편 — 이 편들의 수용률은 못 믿는다: {untrusted[:5]}")
    if skipped:
        print(f"!! 건너뛴 {len(skipped)}편: {skipped[:3]}")
    print("거부 사유:", reason_counts if reason_counts else "없음")
    margin_summary(results)                       # [joint_margin_patch]
    print(f"→ {a.out}")
    print("⚠️ simulation-only episode-start alignment 기준이다. 실물 캘리브레이션이 아니다.")


if __name__ == "__main__":
    main()
