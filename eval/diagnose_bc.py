"""Why does a policy with val_loss 0.005 score 0% at rollout?
val_loss 0.005 인 정책이 롤아웃 0% 인 이유는 무엇인가?

A low imitation loss and a zero success rate are not a contradiction — they are
the textbook symptom of a policy that learned the easiest function that fits the
data. In a scripted pick demonstration most timesteps ask the arm to stay very
near where it already is, so "output the current joint angles" fits almost every
frame. That function has a tiny L1 loss and moves the robot exactly nowhere.
낮은 모방 손실과 0% 성공률은 모순이 아니다. 데이터에 맞는 가장 쉬운 함수를
학습한 정책의 교과서적 증상이다. 스크립트 픽 시연에서 대부분의 스텝은 팔에게
"지금 있는 자리 근처에 있어라"를 요구한다. 그래서 "현재 관절각을 그대로 출력"이
거의 모든 프레임에 들어맞는다. 그 함수의 L1 손실은 아주 작고, 로봇은 조금도
움직이지 않는다.

So the diagnostic is not "is the loss low" but "is the loss lower than a function
that cannot possibly work". Two such functions are scored here on the same
frames as the policy:
따라서 진단은 "손실이 낮은가"가 아니라 "**작동할 수 없는 함수보다** 낮은가"다.
그런 함수 둘을 정책과 같은 프레임에서 채점한다.

  identity   action = 현재 state   ("가만히 있어라")
  mean       action = 데이터셋 평균 행동   ("항상 같은 자세")

If the policy does not clearly beat both, its loss says nothing about whether it
can do the task, and no amount of further training on this data will help.
정책이 둘을 뚜렷하게 이기지 못하면, 그 손실은 태스크 수행 가능성에 대해 아무것도
말하지 않는다. 이 데이터로 더 학습해도 달라지지 않는다.

⚠️ 이 검사는 롤아웃을 대체하지 않는다. 여기를 통과해도 실제 성공률은 별개다.
   여기서 실패하면 롤아웃을 볼 필요가 없다는 것까지만 말한다.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contract.episode import read_episode
from policy.bc import BCPolicy
from policy.act import load_policy
from sim.base import Observation
from sim.mujoco.build_scene import DEFAULT_CONFIG, joint_specs, load_config
from tracking.exp_log import file_digest, log_run

# Fixed before the numbers were computed.
# 수치를 계산하기 전에 확정했다.
GATES: dict[str, str] = {
    "beats_trivial": (
        "BC 의 L1 <= 0.5 x (identity, mean 중 더 낮은 값). 못 넘으면 손실이 낮은 이유는 "
        "태스크를 배워서가 아니라 태스크가 쉬운 함수로 근사되기 때문이다."
    ),
    "not_collapsed": (
        "BC 예측 행동의 표준편차 >= 정답 행동 표준편차의 0.5배. 못 넘으면 모드 평균으로 "
        "붕괴한 것이다 — 어떤 관측에도 거의 같은 행동을 낸다."
    ),
}
BEAT_RATIO = 0.5
STD_RATIO = 0.5


@dataclass
class PredictorScore:
    """One predictor's error over the evaluated frames.
    예측기 하나의 평가 프레임 전체 오차."""

    name: str
    l1: float
    per_joint_l1: list[float]
    per_joint_deg: list[float]
    pred_std: list[float]


def _to_degrees(delta_norm: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Convert a normalised difference into degrees of joint travel.
    정규화 차이를 관절 이동 각도(도)로 바꾼다."""
    specs = joint_specs(cfg)
    half = np.array([(s.hi - s.lo) / 2.0 for s in specs], dtype=np.float64)
    return np.degrees(delta_norm * half)


def evaluate(
    dataset: Path, ckpt: Path, *, limit: int, device: str
) -> tuple[list[PredictorScore], dict[str, Any]]:
    """Score the policy and the two trivial predictors on the same frames.
    정책과 두 자명한 예측기를 같은 프레임에서 채점한다."""
    cfg = load_config()
    files = sorted(dataset.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"에피소드가 없다: {dataset}")
    # The tail of the directory is what a sequential split holds out, so scoring
    # there is closer to held-out than scoring the head would be.
    # 순차 분할은 디렉터리 뒤쪽을 남긴다. 그래서 뒤쪽에서 채점하는 편이 앞쪽보다
    # held-out 에 가깝다.
    files = files[-limit:] if limit > 0 else files

    policy = load_policy(ckpt, device=device)
    print(f"{policy.describe()}\n")
    print(f"평가 대상 {len(files)}편 (디렉터리 뒤쪽), 장치 {device}")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    preds: list[np.ndarray] = []

    for path in files:
        ep = read_episode(path)
        for t in range(ep.meta.n_steps):
            obs = Observation(
                images={c: ep.images[c][t] for c in ep.meta.cameras},
                state=ep.state[t],
                timestamp=float(ep.state_timestamp[t]),
            )
            preds.append(policy.act(obs))
            states.append(ep.state[t])
            actions.append(ep.action[t])

    S = np.asarray(states, dtype=np.float64)
    A = np.asarray(actions, dtype=np.float64)
    P = np.asarray(preds, dtype=np.float64)
    mean_action = A.mean(axis=0)

    def score(name: str, hat: np.ndarray) -> PredictorScore:
        err = np.abs(hat - A)
        per_joint = err.mean(axis=0)
        return PredictorScore(
            name=name,
            l1=float(err.mean()),
            per_joint_l1=[float(v) for v in per_joint],
            per_joint_deg=[float(v) for v in _to_degrees(per_joint, cfg)],
            pred_std=[float(v) for v in hat.std(axis=0)],
        )

    scores = [
        score("bc", P),
        score("identity", S),
        score("mean", np.broadcast_to(mean_action, A.shape)),
    ]

    # How much the demonstration actually asks the arm to move each tick. If this
    # is near zero the task is mostly "hold still" and the loss is uninformative.
    # 시연이 매 틱 팔에게 실제로 요구하는 이동량. 0 에 가까우면 태스크가 대부분
    # "가만히 있기"이고 손실은 아무것도 알려주지 않는다.
    step_move = np.abs(A - S).mean(axis=0)

    extra = {
        "n_frames": int(A.shape[0]),
        "n_episodes": len(files),
        "gt_action_std": [float(v) for v in A.std(axis=0)],
        "action_minus_state_norm": [float(v) for v in step_move],
        "action_minus_state_deg": [float(v) for v in _to_degrees(step_move, cfg)],
        "clip_report": policy.clip_report(),
    }
    return scores, extra


def format_report(
    scores: list[PredictorScore], extra: dict[str, Any], cfg: dict[str, Any]
) -> tuple[str, dict[str, bool]]:
    """The table for the MEASURE document, plus the gate verdicts.
    MEASURE 문서용 표와 게이트 판정."""
    names = [s.name for s in joint_specs(cfg)]
    by = {s.name: s for s in scores}
    lines: list[str] = []

    lines.append(f"{'예측기':10s} {'L1 (정규화)':>14s}   설명")
    lines.append("-" * 62)
    desc = {
        "bc": "학습된 정책",
        "identity": "현재 자세 그대로 (가만히 있기)",
        "mean": "항상 데이터셋 평균 자세",
    }
    for s in scores:
        lines.append(f"{s.name:10s} {s.l1:14.6f}   {desc[s.name]}")

    lines.append("")
    lines.append(f"{'관절':15s} {'BC 오차(도)':>12s} {'identity(도)':>13s} "
                 f"{'시연 이동량(도)':>16s}")
    lines.append("-" * 62)
    for i, jn in enumerate(names):
        lines.append(
            f"{jn:15s} {by['bc'].per_joint_deg[i]:12.4f} "
            f"{by['identity'].per_joint_deg[i]:13.4f} "
            f"{extra['action_minus_state_deg'][i]:16.4f}"
        )

    lines.append("")
    lines.append(f"{'관절':15s} {'BC 예측 표준편차':>18s} {'정답 표준편차':>15s} {'비율':>8s}")
    lines.append("-" * 62)
    for i, jn in enumerate(names):
        gt = extra["gt_action_std"][i]
        pr = by["bc"].pred_std[i]
        ratio = pr / gt if gt > 1e-12 else float("nan")
        lines.append(f"{jn:15s} {pr:18.6f} {gt:15.6f} {ratio:8.3f}")

    trivial_best = min(by["identity"].l1, by["mean"].l1)
    beats = by["bc"].l1 <= BEAT_RATIO * trivial_best
    ratios = [
        by["bc"].pred_std[i] / extra["gt_action_std"][i]
        for i in range(len(names))
        if extra["gt_action_std"][i] > 1e-12
    ]
    not_collapsed = bool(ratios) and min(ratios) >= STD_RATIO

    lines.append("")
    lines.append(f"{extra['clip_report']}")
    lines.append("")
    lines.append("게이트 판정:")
    lines.append(
        f"  [beats_trivial] bc {by['bc'].l1:.6f} <= {BEAT_RATIO} x {trivial_best:.6f} "
        f"= {BEAT_RATIO * trivial_best:.6f} → {'통과' if beats else '**실패**'}"
    )
    lines.append(
        f"  [not_collapsed] 최소 표준편차 비율 {min(ratios) if ratios else float('nan'):.3f} "
        f">= {STD_RATIO} → {'통과' if not_collapsed else '**실패 — 모드 평균 붕괴**'}"
    )
    return "\n".join(lines), {"beats_trivial": beats, "not_collapsed": not_collapsed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/sim_pick_v1"))
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/bc/bc_sim_pick_v1.pt"))
    parser.add_argument("--limit", type=int, default=20, help="뒤쪽 N편만 평가 (0=전부)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg = load_config()

    print("게이트 기준 (결과 확인 전 확정):")
    for key, text in GATES.items():
        print(f"  [{key}] {text}")
    print()

    scores, extra = evaluate(
        args.dataset, args.ckpt, limit=args.limit, device=args.device
    )
    report, verdicts = format_report(scores, extra, cfg)
    print()
    print(report)

    passed = all(verdicts.values())
    print()
    if passed:
        print(
            "손실이 자명한 예측기보다 뚜렷하게 낮다. 즉 0% 의 원인은 학습 목표가 아니라 "
            "다른 곳에 있다 — 오차 누적, 관측 분포 차이, 또는 제어 주기를 봐야 한다."
        )
    else:
        print(
            "손실이 낮은 이유가 태스크 학습이 아니다. 이 데이터·이 목표로는 더 학습해도 "
            "롤아웃이 개선되지 않는다. 데이터(궤적 다양성)나 목표(행동 청킹·델타 행동)를 "
            "바꿔야 한다."
        )
    print("\n⚠️ 이 검사는 롤아웃을 대체하지 않는다. 통과해도 성공률은 별개다.")

    if args.log:
        rec = log_run(
            experiment="diagnose_bc",
            author=args.author,
            issue="S15P21A103-34",
            conditions={
                "dataset": str(args.dataset),
                "ckpt": str(args.ckpt),
                "limit": args.limit,
                "device": args.device,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "gates": GATES,
            },
            result={
                "scores": [s.__dict__ for s in scores],
                "verdicts": verdicts,
                **extra,
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
