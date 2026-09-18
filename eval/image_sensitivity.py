"""Does the policy's action actually depend on what it sees?
정책의 행동이 실제로 보는 것에 따라 달라지는가?

Why this is the question to settle first.
왜 이걸 먼저 가려야 하는가.

A vision policy takes (image, state) and returns an action. If the image
contributes nothing, the policy is a function of joint angles alone — it replays
an average trajectory and cannot find an object that moved. That failure looks
identical to "not accurate enough" from the outside: both score 0%. But the
fixes are opposite. Better action representation and loss weighting help a
policy that responds imprecisely; they do nothing for a policy that is not
responding at all.
시각 정책은 (이미지, 상태)를 받아 행동을 낸다. 이미지가 아무 기여도 하지 않으면
그 정책은 관절각만의 함수이고, 평균 궤적을 재생할 뿐 움직인 물체를 찾지 못한다.
이 실패는 밖에서 보면 "정밀도가 부족하다"와 똑같이 생겼다 — 둘 다 0% 다. 그런데
대책이 정반대다. 행동 표현과 손실 가중은 부정확하게 반응하는 정책을 돕지,
아예 반응하지 않는 정책에는 아무것도 하지 못한다.

How it is measured — swap one input, hold the other.
어떻게 재는가 — 한 입력만 바꾸고 나머지는 고정한다.

At the same timestep of two different episodes the object sits in a different
place, so the images differ and the recorded actions differ. Feeding episode i's
state with episode j's image isolates the image's contribution: whatever the
output moves by came from pixels alone.
서로 다른 두 에피소드의 같은 시점에서는 물체가 다른 자리에 있으므로 이미지가 다르고
기록된 행동도 다르다. 에피소드 i 의 상태에 에피소드 j 의 이미지를 넣으면 이미지의
기여만 분리된다. 출력이 움직인 만큼이 전부 픽셀에서 온 것이다.

⚠️ 교체된 (이미지, 상태) 쌍은 실제로 함께 나타난 적 없는 조합이다. 절대값을
   물리량으로 읽지 마라. 읽어야 할 것은 **비율**이다 — 정답 행동이 물체 위치에
   따라 달라지는 폭 대비 얼마나 달라지는가.
"""

from __future__ import annotations

import argparse
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
GATE_IMAGE_RATIO = 0.30
GATES: dict[str, str] = {
    "responds_to_image": (
        f"이미지 교체가 만드는 행동 변화 >= 정답 행동의 에피소드 간 변화 x {GATE_IMAGE_RATIO}. "
        "못 넘으면 정책이 이미지를 사실상 무시하는 것이다. 분모는 체크포인트의 회귀 공간을 따른다."
    ),
}


@dataclass
class Sensitivity:
    """How much the output moves when one input is swapped.
    한 입력을 교체했을 때 출력이 얼마나 움직이는가."""

    per_joint_image: list[float]
    per_joint_state: list[float]
    per_joint_gt: list[float]
    # The same between-episode difference, but of the residual `action - state`.
    # 같은 에피소드 간 차이를, 잔차 `action - state` 에 대해 잰 것.
    #
    # This is the number that decides whether a delta action target helps. With an
    # absolute target, the between-episode difference in `action` equals the
    # difference in `state` -- the object moved, so the arm is elsewhere, and that
    # displacement IS the answer. The network reads it off the state and the image
    # has nothing left to explain. Subtracting the state removes exactly that
    # shortcut, and what remains is what vision would have to supply.
    # 델타 행동 목표가 도움이 되는지를 결정하는 수치다. 절대 목표에서는 에피소드 간
    # `action` 차이가 곧 `state` 차이다. 물체가 옮겨졌으니 팔도 다른 곳에 있고, 그
    # 변위가 곧 답이다. 신경망은 그걸 상태에서 읽으면 되고 이미지가 설명할 몫은 남지
    # 않는다. 상태를 빼면 정확히 그 지름길이 사라지고, 남는 것이 시각이 공급해야 할
    # 몫이다.
    per_joint_gt_delta: list[float]
    n_pairs: int
    n_timesteps: int
    # Which space the checkpoint regresses -- decides the gate's denominator.
    # 체크포인트가 회귀하는 공간. 게이트의 분모를 정한다.
    action_space: str = "joint_absolute"

    def ratio(self) -> list[float]:
        return [
            (img / gt) if gt > 1e-12 else float("nan")
            for img, gt in zip(self.per_joint_image, self.per_joint_gt)
        ]

    def overall_ratio(self) -> float:
        img = float(np.mean(self.per_joint_image))
        gt = float(np.mean(self.per_joint_gt))
        return img / gt if gt > 1e-12 else float("nan")

    def delta_ratio(self) -> float:
        """Image contribution measured against the residual, not the absolute target.
        절대 목표가 아니라 잔차를 기준으로 잰 이미지 기여도."""
        img = float(np.mean(self.per_joint_image))
        gt = float(np.mean(self.per_joint_gt_delta))
        return img / gt if gt > 1e-12 else float("nan")

    def state_explains(self) -> float:
        """How much of the absolute target the state alone accounts for.
        절대 목표 중 상태만으로 설명되는 몫."""
        st = float(np.mean(self.per_joint_state))
        gt = float(np.mean(self.per_joint_gt))
        return st / gt if gt > 1e-12 else float("nan")

    def gate_ratio(self, action_space: str, gripper_idx: int) -> tuple[str, float]:
        """The image/ground-truth ratio in the space the network actually regresses.
        신경망이 실제로 회귀하는 공간에서의 이미지/정답 비율.

        The denominator must be the between-episode variation of *what the network
        outputs*. For an absolute policy that is the absolute action difference; for a
        delta policy it is the delta difference, because the state term is added back
        after the network and cannot depend on the image. Measuring a delta policy
        against the absolute reference reports the state's share, not the image's —
        it printed 0.033 for a policy whose arm output moved 0.68~0.96x the delta
        reference under image swaps. Same threshold, correct denominator.
        분모는 **신경망이 출력하는 것**의 에피소드 간 변화여야 한다. 절대 정책이면 절대
        행동 차이, 델타 정책이면 델타 차이다 — state 항은 신경망 뒤에서 더해지므로
        이미지에 의존할 수 없다. 델타 정책을 절대 기준으로 재면 이미지 몫이 아니라 상태
        몫이 나온다. 이미지 교체에 팔 출력이 델타 기준의 0.68~0.96배 움직인 정책에
        0.033 을 찍었다. 기준값은 그대로, 분모만 바로잡았다.

        When the gripper is regressed absolutely its ground truth does not vary
        across episodes at matched ticks (scripted timing), so it is excluded from the
        ratio and reported on its own.
        그리퍼를 절대로 회귀하면 같은 틱의 정답이 에피소드 간에 변하지 않으므로(스크립트
        타이밍) 비율에서 빼고 따로 보고한다.
        """
        img = np.asarray(self.per_joint_image, dtype=float)
        if action_space == "joint_absolute":
            gt = np.asarray(self.per_joint_gt, dtype=float)
            label = "정답 절대차"
        else:
            gt = np.asarray(self.per_joint_gt_delta, dtype=float)
            label = "정답 델타차"
        if action_space == "joint_delta_gripper_abs":
            keep = [i for i in range(len(img)) if i != gripper_idx]
            img, gt = img[keep], gt[keep]
            label += "(팔 5관절)"
        g = float(gt.mean())
        return label, (float(img.mean()) / g if g > 1e-12 else float("nan"))


def _obs(images: dict[str, np.ndarray], state: np.ndarray, t: float) -> Observation:
    return Observation(images=images, state=state, timestamp=t)


def measure(
    dataset: Path,
    ckpt: Path,
    *,
    n_episodes: int,
    stride: int,
    device: str,
) -> Sensitivity:
    """Swap images between episodes at matching timesteps and watch the output.
    같은 시점에서 에피소드 간 이미지를 교체하고 출력을 본다."""
    files = sorted(dataset.glob("*.npz"))[:n_episodes]
    if len(files) < 2:
        raise SystemExit(f"에피소드가 2편 이상 필요하다: {dataset}")

    policy = load_policy(ckpt, device=device)
    print(f"{policy.describe()}\n")

    eps = [read_episode(p) for p in files]
    cams = eps[0].meta.cameras
    T = min(e.meta.n_steps for e in eps)
    steps = list(range(0, T, max(1, stride)))
    print(f"에피소드 {len(eps)}편 · 시점 {len(steps)}개 (stride {stride}) · 장치 {device}")

    d_image: list[np.ndarray] = []
    d_state: list[np.ndarray] = []
    d_gt: list[np.ndarray] = []
    d_gt_delta: list[np.ndarray] = []

    for t in steps:
        for i in range(len(eps)):
            ei = eps[i]
            base = policy.act(
                _obs({c: ei.images[c][t] for c in cams}, ei.state[t], float(ei.state_timestamp[t]))
            )
            for j in range(len(eps)):
                if i == j:
                    continue
                ej = eps[j]
                # 이미지만 교체: 상태는 i, 픽셀은 j
                swap_img = policy.act(
                    _obs({c: ej.images[c][t] for c in cams}, ei.state[t],
                         float(ei.state_timestamp[t]))
                )
                # 상태만 교체: 픽셀은 i, 상태는 j
                swap_state = policy.act(
                    _obs({c: ei.images[c][t] for c in cams}, ej.state[t],
                         float(ei.state_timestamp[t]))
                )
                d_image.append(np.abs(swap_img - base))
                d_state.append(np.abs(swap_state - base))
                d_gt.append(np.abs(ej.action[t] - ei.action[t]))
                d_gt_delta.append(
                    np.abs(
                        (ej.action[t] - ej.state[t]) - (ei.action[t] - ei.state[t])
                    )
                )

    return Sensitivity(
        per_joint_image=[float(v) for v in np.mean(d_image, axis=0)],
        per_joint_state=[float(v) for v in np.mean(d_state, axis=0)],
        per_joint_gt=[float(v) for v in np.mean(d_gt, axis=0)],
        per_joint_gt_delta=[float(v) for v in np.mean(d_gt_delta, axis=0)],
        n_pairs=len(d_image),
        n_timesteps=len(steps),
        action_space=policy.action_space,
    )


def format_report(s: Sensitivity, cfg: dict[str, Any]) -> str:
    names = [j.name for j in joint_specs(cfg)]
    lines = [
        f"{'관절':15s} {'이미지 교체':>12s} {'상태 교체':>12s} {'정답 차이':>12s} "
        f"{'이미지/정답':>12s} {'정답 델타차':>12s} {'이미지/델타':>12s}",
        "-" * 96,
    ]
    for i, jn in enumerate(names):
        gtd = s.per_joint_gt_delta[i]
        rd = s.per_joint_image[i] / gtd if gtd > 1e-12 else float("nan")
        lines.append(
            f"{jn:15s} {s.per_joint_image[i]:12.6f} {s.per_joint_state[i]:12.6f} "
            f"{s.per_joint_gt[i]:12.6f} {s.ratio()[i]:12.3f} {gtd:12.6f} {rd:12.3f}"
        )
    lines.append("-" * 96)
    lines.append(
        f"{'전체':15s} {np.mean(s.per_joint_image):12.6f} "
        f"{np.mean(s.per_joint_state):12.6f} {np.mean(s.per_joint_gt):12.6f} "
        f"{s.overall_ratio():12.3f} {np.mean(s.per_joint_gt_delta):12.6f} "
        f"{s.delta_ratio():12.3f}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/sim_pick_v1"))
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/bc/bc_sim_pick_v1.pt"))
    parser.add_argument("--episodes", type=int, default=6, help="교체에 쓸 에피소드 수")
    parser.add_argument("--stride", type=int, default=10, help="몇 틱마다 잴 것인가")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg = load_config()

    print("게이트 기준 (결과 확인 전 확정):")
    for k, v in GATES.items():
        print(f"  [{k}] {v}")
    print()

    s = measure(
        args.dataset, args.ckpt,
        n_episodes=args.episodes, stride=args.stride, device=args.device,
    )
    print()
    print(format_report(s, cfg))
    print(f"\n교체 쌍 {s.n_pairs}개")

    from policy.bc import gripper_index

    space = s.action_space
    g_idx = gripper_index()
    label, ratio = s.gate_ratio(space, g_idx)
    print("\n게이트 판정:")
    print(
        f"  [responds_to_image] 이미지/{label} {ratio:.3f} >= {GATE_IMAGE_RATIO:.2f} → "
        f"{'통과 — 정책 출력이 이미지에 반응한다' if ratio >= GATE_IMAGE_RATIO else '**실패 — 정책이 이미지를 사실상 무시한다**'}"
    )
    print(f"  회귀 공간 {space} 에 맞는 분모를 썼다. 참고: 절대 기준 {s.overall_ratio():.3f} · "
          f"델타 기준(전 관절) {s.delta_ratio():.3f} · 상태가 설명하는 절대 목표 몫 {s.state_explains():.3f}")
    if space == "joint_delta_gripper_abs":
        gi, gs, gg = s.per_joint_image[g_idx], s.per_joint_state[g_idx], s.per_joint_gt[g_idx]
        print(f"  그리퍼(절대 회귀): 이미지 교체 {gi:.6f} · 상태 교체 {gs:.6f} · 같은 틱 정답 차이 {gg:.6f}"
              " — 정답이 틱에 고정돼 비율을 정의할 수 없다. 출력 흔들림 크기만 기록한다")
    print("  ⚠️ 이 도구는 민감도를 잰다. 정확도가 아니다 — 반응한다고 맞게 반응하는 것은 아니다")
    print()

    if args.log:
        rec = log_run(
            experiment="image_sensitivity",
            author=args.author,
            issue="S15P21A103-34",
            conditions={
                "dataset": str(args.dataset),
                "ckpt": str(args.ckpt),
                "episodes": args.episodes,
                "stride": args.stride,
                "device": args.device,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "gates": GATES,
                "method": "같은 시점에서 에피소드 간 이미지/상태를 교체하고 출력 변화를 측정",
            },
            result={
                "per_joint_image": s.per_joint_image,
                "per_joint_state": s.per_joint_state,
                "per_joint_gt": s.per_joint_gt,
                "ratio_per_joint": s.ratio(),
                "overall_ratio": s.overall_ratio(),
                "gate_ratio": ratio,
                "gate_reference": label,
                "action_space": space,
                "per_joint_gt_delta": s.per_joint_gt_delta,
                "delta_ratio": s.delta_ratio(),
                "state_explains": s.state_explains(),
                "passed": bool(ratio >= GATE_IMAGE_RATIO),
                "n_pairs": s.n_pairs,
                "n_timesteps": s.n_timesteps,
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")

    return 0 if ratio >= GATE_IMAGE_RATIO else 1


if __name__ == "__main__":
    raise SystemExit(main())
