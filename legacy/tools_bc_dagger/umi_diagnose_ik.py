"""M8-a 를 좁힌다 — 도달 가능한 pose 에서 IK 가 못 푸는 0.17% 는 왜인가.
Narrowing M8-a: why does IK fail on 0.17% of poses that are reachable by construction.

S15P21A103-127 · MEASURE_umi_roundtrip_0907 M8-a 후속.

입력은 유효한 관절 배치에서 FK 로 만든 것이라 **전부 도달 가능**하다. 그런데
1800 스텝 중 3 이 미수렴이다. 실데이터에서는 이 비율이 훨씬 높아지고, 그러면
수용률의 주요 손실원이 된다. 원인을 지금 좁혀둔다.

## 가설 두 개와 그것을 구분하는 계측

**H1 특이자세.** 그 자세에서 구속 야코비안이 랭크를 잃어 감쇠 최소자승이
   내려가지 못한다 → 최소 특이값이 성공 스텝보다 뚜렷하게 작아야 한다
**H2 시딩/분지.** 직전 해에서 출발해 다른 분지에 갇혔다
   → `q_init` 을 참값으로 주면 수렴해야 한다

두 계측이 서로를 배제한다. H1 이면 참값 시딩으로도 못 풀고, H2 면 풀린다.

    # [로컬]
    cd AI && python tools/umi_diagnose_ik.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from paths import DEFAULT_CONFIG, DEFAULT_SCENE  # noqa: E402
from sim.mujoco.build_scene import build_model, load_config  # noqa: E402
from sim.mujoco.kinematics import IKResult, grasp_point  # noqa: E402
from tools.make_umi_synth import gripper_profile  # noqa: E402
from tools.umi_mujoco import (  # noqa: E402
    GRIPPER_BODY,
    MujocoIK,
    N_JOINTS,
    eef_pose_from_joints,
    gap_from_angle,
    joint_ranges,
    smooth_joint_trajectory,
)

FREE = [0, 1, 2, 3]  # wrist_roll 은 match_roll 이 따로 정한다
AXIS_WEIGHT = 0.15   # solve_pose_ik 의 기본값


def constraint_sigma_min(model, data, q, pinch) -> float:
    """Smallest singular value of the constraint Jacobian at a configuration.
    주어진 자세에서 구속 야코비안의 최소 특이값.

    Same stack `solve_pose_ik` builds: position rows plus axis rows scaled by
    `axis_weight`, restricted to the joints the solve is allowed to move.
    `solve_pose_ik` 가 쌓는 것과 같은 행렬이다 — 위치 행 + `axis_weight` 로
    가중한 축 행, 풀이가 움직여도 되는 관절로 제한.
    """
    data.qpos[:N_JOINTS] = q
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
    point = grasp_point(model, data, pinch)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, point, bid)
    jac = np.vstack([jacp[:, FREE], AXIS_WEIGHT * jacr[:, FREE]])
    return float(np.linalg.svd(jac, compute_uv=False).min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0, help="make_umi_synth 와 같은 시드여야 재현된다")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    ik = MujocoIK(model, cfg)
    data = mujoco.MjData(model)
    ranges = joint_ranges(cfg)
    pinch = np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float)
    curve = cfg["grasp"]["gap_curve"]
    rng = np.random.default_rng(args.seed)

    sig_ok: list[float] = []
    sig_bad: list[float] = []
    rows = []

    for e in range(args.episodes):
        q_true = smooth_joint_trajectory(ranges, args.steps, rng)
        q_true[:, 5] = gripper_profile(args.steps, ranges[5, 0], ranges[5, 1])
        poses = [eef_pose_from_joints(model, data, row, pinch) for row in q_true]
        prev = None
        for i, ((pos, quat), qref) in enumerate(zip(poses, q_true)):
            sol = ik.solve(pos, quat, gap_from_angle(qref[5], curve), q_init=prev)
            sigma = constraint_sigma_min(model, data, qref, pinch)
            if sol.converged and sol.within_limits:
                sig_ok.append(sigma)
            else:
                sig_bad.append(sigma)
                # H2: 참값으로 시딩하면 풀리는가
                reseed = ik.solve(pos, quat, gap_from_angle(qref[5], curve), q_init=qref)
                lim = float(np.min(np.minimum(qref[:5] - ranges[:5, 0], ranges[:5, 1] - qref[:5])))
                rows.append({
                    "ep": e, "step": i, "boundary": i in (0, args.steps - 1),
                    "pos_mm": sol.pos_error_m * 1000, "axis_deg": sol.axis_error_deg,
                    "sigma": sigma, "limit_margin_rad": lim,
                    "reseed_pos_mm": reseed.pos_error_m * 1000,
                    "reseed_axis_deg": reseed.axis_error_deg,
                    "reseed_ok": reseed.converged and reseed.within_limits,
                })
            # `umi.convert.convert` 와 **같은** 시드 정책이어야 한다 — 실패한 해는
            # 다음 스텝에 넘기지 않는다. 처음엔 무조건 넘겼고, 그래서 나쁜 시드가
            # 연쇄해 실패가 3건에서 11건으로 불어났다. 진단기가 파이프라인과
            # 다르게 동작하면 진단기의 수치를 재는 것이 된다.
            if sol.converged and sol.within_limits:
                prev = sol.q_rad

    n = args.episodes * args.steps
    print(f"n={n} · 미수렴/한계 {len(sig_bad)}건 ({len(sig_bad)/n*100:.2f}%)\n")
    if not rows:
        print("실패 스텝이 없다. 시드나 인자가 make_umi_synth 실행과 다르지 않은지 확인해라")
        return 1

    print("실패 스텝 상세")
    print(f"{'ep':>3}{'step':>6}{'끝':>4}{'위치mm':>10}{'축도':>9}{'σmin':>10}"
          f"{'한계여유rad':>12}{'참값시딩 위치mm':>16}{'참값시딩 축도':>14}{'수렴':>6}")
    for r in rows:
        print(f"{r['ep']:>3}{r['step']:>6}{'Y' if r['boundary'] else '-':>4}"
              f"{r['pos_mm']:>10.3f}{r['axis_deg']:>9.3f}{r['sigma']:>10.5f}"
              f"{r['limit_margin_rad']:>12.4f}{r['reseed_pos_mm']:>16.4f}"
              f"{r['reseed_axis_deg']:>14.4f}{'Y' if r['reseed_ok'] else 'N':>6}")

    ok = np.array(sig_ok); bad = np.array(sig_bad)
    print(f"\nσmin 분포")
    print(f"  성공 {len(ok):4d}건  중앙 {np.median(ok):.5f}  5퍼센타일 {np.percentile(ok,5):.5f}  최소 {ok.min():.5f}")
    print(f"  실패 {len(bad):4d}건  중앙 {np.median(bad):.5f}  최대 {bad.max():.5f}  최소 {bad.min():.5f}")
    n_below = int((ok < bad.max()).sum())
    print(f"  성공 스텝 중 실패 최대 σmin({bad.max():.5f}) 보다 작은 것: {n_below}건 "
          f"({n_below/len(ok)*100:.2f}%)")

    print("\n=== 가설 판정 ===")
    reseed_fixed = sum(r["reseed_ok"] for r in rows)
    h1 = len(bad) > 0 and bad.max() < np.percentile(ok, 5)
    print(f"H1 특이자세  — 실패 σmin 최대 < 성공 5퍼센타일 ?  "
          f"{bad.max():.5f} < {np.percentile(ok,5):.5f} → {'지지' if h1 else '기각'}")
    print(f"H2 시딩/분지 — 참값 시딩으로 수렴 ?  {reseed_fixed}/{len(rows)} → "
          f"{'지지' if reseed_fixed == len(rows) else ('부분' if reseed_fixed else '기각')}")
    n_boundary = sum(r["boundary"] for r in rows)
    print(f"\n참고: 실패 {len(rows)}건 중 에피소드 끝에 있는 것 {n_boundary}건 "
          f"(끝이 아니면 최장 연속 구간 규칙으로 에피소드가 크게 잘린다 — M8-b)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
