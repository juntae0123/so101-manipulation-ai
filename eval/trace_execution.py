"""Split a 0% rollout into its three possible causes, and name which one it is.
0% 롤아웃을 가능한 세 원인으로 쪼개고, 그중 어느 것인지 이름을 붙인다.

A policy that memorised its single training episode (train loss 0.7% of the
trivial predictor) and still lifts 0.00 cm at the exact recorded object position
has failed for one of three reasons, and they need different fixes:

  A 실행 경로   the recorded action sequence itself does not reproduce the
                episode when fed back into the environment. Then no policy can
                succeed here and nothing about learning is at fault.
  B 네트워크    given the recorded observation, the network does not output the
                recorded action. Then the memorisation did not transfer to
                inference (dtype, normalisation, eval-mode, checkpoint).
  C 폐루프      the network reproduces recorded actions on recorded inputs, but
                the rollout's own observations differ from the recorded ones,
                so it is off-manifold from some step onward.

세 원인은 처방이 서로 다르다. 이 도구는 판정만 낸다 — 처방은 판정 뒤에 나온다.

Instrument verification: section A is itself the check on the instrument. If
replaying the episode's own actions fails, sections B and C are unreadable and
are reported as void.
계측기 검증: A 절이 계측기 자체의 검사다. 자기 행동 재생이 실패하면 B·C 절은
읽을 수 없으므로 무효로 보고한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from contract.episode import read_episode
from policy.bc import BCPolicy
from policy.act import load_policy
from sim.base import Observation
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config
from sim.mujoco.env import MujocoPickEnv
from tracking.exp_log import file_digest, log_run


# Fixed before any number was produced.
# 어떤 수치도 나오기 전에 확정했다.
GATES: dict[str, str] = {
    "A_replay": "기록된 행동을 그대로 되먹이면 성공해야 한다. 실패하면 B·C 는 무효다.",
    "B_teacher": "기록 관측을 넣었을 때 예측 오차 평균 / 기록 델타 평균 < 0.30",
    "C_obs_t0": "롤아웃 첫 관측과 기록 첫 관측이 일치해야 한다 "
                "(state 최대차 < 1e-3, image 평균차 < 2.0 계조)",
}
B_RATIO_MAX = 0.30
C_STATE_MAX = 1e-3
C_IMAGE_MAX = 2.0


def _obs_at(ep: Any, t: int) -> Observation:
    """Rebuild the observation the collector recorded at tick `t`.
    수집기가 틱 t 에 기록한 관측을 그대로 되만든다."""
    return Observation(
        images={cam: arr[t] for cam, arr in ep.images.items()},
        state=ep.state[t].astype(np.float32),
        timestamp=float(ep.state_timestamp[t]),
    )


def section_a(env: MujocoPickEnv, ep: Any, xy: tuple[float, float]) -> dict[str, Any]:
    """Feed the episode's own recorded actions back into the environment.
    에피소드 자신의 기록 행동을 환경에 그대로 되먹인다."""
    env.reset(object_xy=xy)
    drift: list[float] = []
    success = False
    for t in range(len(ep.action)):
        obs = env.step(ep.action[t])
        drift.append(float(np.max(np.abs(obs.state - ep.state[min(t + 1, len(ep.state) - 1)]))))
        if env.is_success():
            success = True
            break
    return {
        "success": success,
        "ticks": t + 1,
        "lift_cm": round(env.lift_height() * 100, 3),
        "state_drift_max": round(max(drift), 6),
        "state_drift_final": round(drift[-1], 6),
    }


def section_b(policy: BCPolicy, ep: Any) -> dict[str, Any]:
    """Ask the network for an action at every recorded observation.
    기록된 모든 관측에서 네트워크에 행동을 물어본다."""
    err = np.zeros(6, dtype=np.float64)
    delta = np.zeros(6, dtype=np.float64)
    n = len(ep.action)
    for t in range(n):
        pred = policy.act(_obs_at(ep, t))
        err += np.abs(pred - ep.action[t])
        delta += np.abs(ep.action[t] - ep.state[t])
    err /= n
    delta /= n
    ratio = float(err.sum() / delta.sum()) if delta.sum() > 0 else float("inf")
    return {
        "n": n,
        "err_per_joint": [round(float(v), 6) for v in err],
        "delta_per_joint": [round(float(v), 6) for v in delta],
        "err_mean": round(float(err.mean()), 6),
        "delta_mean": round(float(delta.mean()), 6),
        "ratio": round(ratio, 4),
        "clipped": f"{policy.n_clipped}/{policy.n_actions}",
    }


def section_c(
    env: MujocoPickEnv, policy: BCPolicy, ep: Any, xy: tuple[float, float]
) -> dict[str, Any]:
    """Run the closed loop and compare every step against the recording.
    폐루프로 돌리면서 매 스텝을 기록과 대조한다."""
    obs = env.reset(object_xy=xy)
    rows: list[dict[str, Any]] = []
    n = len(ep.action)
    for t in range(n):
        s_diff = float(np.max(np.abs(obs.state - ep.state[t])))
        i_diff = float(
            np.mean([
                np.mean(np.abs(obs.images[c].astype(np.int16) - ep.images[c][t].astype(np.int16)))
                for c in obs.images
            ])
        )
        pred = policy.act(obs)
        a_diff = float(np.max(np.abs(pred - ep.action[t])))
        rows.append({
            "t": t,
            "state_diff": round(s_diff, 6),
            "image_diff": round(i_diff, 4),
            "action_diff": round(a_diff, 6),
        })
        obs = env.step(pred)
        if env.is_success():
            break
    return {"rows": rows, "success": env.is_success(), "lift_cm": round(env.lift_height() * 100, 3)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--policy-ckpt", type=Path, required=True)
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    ep = read_episode(args.episode)
    meta = json.loads(args.episode.with_suffix(".json").read_text(encoding="utf-8"))
    xy = tuple(float(v) for v in meta["notes"]["object_init_xy"])
    cfg = load_config()

    print("게이트 기준 (결과 확인 전 확정):")
    for k, v in GATES.items():
        print(f"  [{k}] {v}")
    print(f"\n대상: {args.episode.name} · {len(ep.action)} 스텝 · 물체 ({xy[0]:+.4f}, {xy[1]:+.4f})")

    policy = load_policy(args.policy_ckpt)
    print(f"정책: {policy.describe()}\n")

    with MujocoPickEnv(cfg, render=True, object_jitter_m=0.0) as env:
        a = section_a(env, ep, xy)
        print("A 실행 경로 — 기록된 행동을 그대로 되먹임")
        print(f"  성공 {a['success']} · {a['ticks']}틱 · 상승 {a['lift_cm']}cm")
        print(f"  기록 state 와의 최대 편차 {a['state_drift_max']} (마지막 {a['state_drift_final']})")
        a_ok = bool(a["success"])
        print(f"  [A_replay] → {'통과 — 실행 경로는 정상이다' if a_ok else '**실패 — 실행 경로가 기록을 재현하지 못한다. 아래 B·C 는 무효**'}\n")

        b = section_b(policy, ep)
        print("B 네트워크 — 기록 관측을 넣었을 때의 예측")
        print(f"  예측오차 평균 {b['err_mean']:.6f} · 기록 델타 평균 {b['delta_mean']:.6f} "
              f"· 비율 {b['ratio']}")
        print(f"  관절별 오차 {b['err_per_joint']}")
        print(f"  관절별 델타 {b['delta_per_joint']}")
        per = [round(e / d, 3) if d > 1e-9 else float("nan")
               for e, d in zip(b["err_per_joint"], b["delta_per_joint"])]
        bad = [i for i, r in enumerate(per) if r == r and r >= B_RATIO_MAX]
        print(f"  관절별 비율(오차÷델타) {per}"
              + (f"  ← {B_RATIO_MAX} 이상: 관절 {bad}" if bad else ""))
        b["ratio_per_joint"] = per
        print(f"  클립 {b['clipped']}")
        b_ok = b["ratio"] < B_RATIO_MAX
        print(f"  [B_teacher] {b['ratio']} < {B_RATIO_MAX} → "
              f"{'통과 — 네트워크는 기록 행동을 재현한다' if b_ok else '**실패 — 추론 시점의 네트워크가 기록 행동을 내지 않는다**'}\n")

        policy.reset()
        c = section_c(env, policy, ep, xy)
        rows = c["rows"]
        print("C 폐루프 — 롤아웃 관측이 기록과 언제 갈라지나")
        print(f"  {'t':>4} {'state차':>10} {'image차':>10} {'action차(max)':>14}")
        print("  (state·action 차는 6관절 중 최대, image 차는 두 카메라 평균 계조. B 절의 관절 평균과 단위가 다르다)")
        idx = sorted({0, 1, 2, 5, 10, 20, 40, 70, len(rows) - 1} & set(range(len(rows))))
        for i in idx:
            r = rows[i]
            print(f"  {r['t']:>4} {r['state_diff']:>10.6f} {r['image_diff']:>10.4f} {r['action_diff']:>14.6f}")
        r0 = rows[0]
        c_ok = r0["state_diff"] < C_STATE_MAX and r0["image_diff"] < C_IMAGE_MAX
        print(f"  [C_obs_t0] state {r0['state_diff']:.6f} < {C_STATE_MAX} · "
              f"image {r0['image_diff']:.4f} < {C_IMAGE_MAX} → "
              f"{'통과 — 첫 관측은 기록과 같다' if c_ok else '**실패 — 롤아웃의 첫 관측부터 기록과 다르다**'}")
        print(f"  폐루프 성공 {c['success']} · 상승 {c['lift_cm']}cm")

    # Judgement only. Which of A/B/C failed, with the numbers that say so. The
    # closed-loop outcome is printed beside it because a B ratio of 0.5 with a
    # successful rollout and a B ratio of 1.03 with the arm frozen are different
    # facts, and the first version of this block called both "nothing transferred".
    # 판정만 낸다. A/B/C 중 무엇이 미달인지와 그 수치. 폐루프 결과를 옆에 두는 이유는
    # B 비율 0.5 에 롤아웃 성공과 B 비율 1.03 에 팔 정지가 다른 사실인데, 이 블록의 첫
    # 버전은 둘을 모두 "넘어오지 않았다"고 불렀기 때문이다.
    print("\n판정:")
    if not a_ok:
        print("  A 미달 — 실행 경로가 기록을 재현하지 못한다. B·C 는 무효.")
    else:
        print("  A 통과 — 실행 경로 정상.")
        if b_ok:
            print(f"  B 통과 — 비율 {b['ratio']}.")
        else:
            print(f"  B 미달 — 비율 {b['ratio']} (기준 {B_RATIO_MAX}), 관절별 {b.get('ratio_per_joint')}."
                  + ("  비율≈1 이면 학습된 것이 추론으로 넘어오지 않은 것이다"
                     if b["ratio"] > 0.9 else "  비율이 1 보다 뚜렷히 낮으면 덜 학습된 것이다"))
        if c_ok:
            print("  C(t=0) 통과 — 첫 관측이 기록과 같다.")
        else:
            print("  C(t=0) 미달 — 롤아웃 관측이 기록과 다르다. 외운 정책이 처음부터 미지 입력을 본다.")
        print(f"  폐루프: {'성공' if c['success'] else '실패'} · 상승 {c['lift_cm']}cm (n=1, 기록 조건 고정)")

    if args.log:
        rec = log_run(
            experiment="trace_execution",
            author=args.author,
            issue="S15P21A103-34",
            conditions={
                "episode": str(args.episode),
                "policy_ckpt": str(args.policy_ckpt),
                "policy_action_space": policy.action_space,
                "object_xy": list(xy),
                "config_sha": file_digest(DEFAULT_CONFIG),
                "gates": GATES,
            },
            result={"A": a, "B": b, "C": {"success": c["success"], "lift_cm": c["lift_cm"], "rows": rows}},
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
