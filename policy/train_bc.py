"""Train a BC policy. Finishes the loop on random tensors before real data exists.
BC 정책을 학습한다. 실데이터가 있기 전에 랜덤 텐서로 루프를 먼저 완주시킨다.

    python tools/train_bc.py --random 256 --epochs 3      # 루프 검증 (데이터 불필요)
    python tools/train_bc.py --data datasets/sim_teleop_v0

⚠️ 이 스크립트는 성능을 판정하지 않는다. 손실만 낸다.
   판정은 `tools/eval_rollout.py --policy-ckpt <ckpt>` 가 하고,
   넘어야 할 값은 롤아웃 성공률 **20.0%** 다 (eval/rollout.py 의 GATES).
   손실 곡선이 내려간 것은 성과가 아니다.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# BLAS 캡(OMP/MKL/...)은 torch 의 intra-op 스레드를 항상 따라가지 않는다.
# 여기서 명시로 건다 — 공유 머신이고, 이 모델은 1.3M 파라미터라 스레드가
# 많아서 빨라지지 않는다. 실측 🟢 2026-09-07: 6개 프로세스에서 load average 165/80.
try:
    import runtime_limits as _rl

    _rl.torch_threads()
except Exception:  # 단독 실행 등 경로에 없을 때
    pass
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from contract.episode import CONTRACT_VERSION
from data.dataset import EpisodeDataset, RandomTensorDataset, collate
from paths import AI_ROOT, DEFAULT_CONFIG
from policy.bc import (
    BCNet,
    CheckpointMeta,
    load_train_config,
    save_checkpoint,
    gripper_index,
    target_scale,
    training_target,
)
from tracking.exp_log import code_digest, file_digest, log_run

DEFAULT_CAMERAS = ["cam_front", "cam_wrist"]


GRIPPER_LOSS_WEIGHT = 1.0
"""Weight on the gripper BCE term relative to the arm regression term.
팔 회귀항 대비 그리퍼 BCE 항의 가중치.

Fixed at 1.0 **before** the experiment ran and not tuned. Choosing it from the
results would be picking a knob to make a number look better, which is what
`PREREG_gripper_binary_head_0910.md` exists to prevent.
실험 착수 **전에** 1.0 으로 고정했고 튜닝하지 않는다. 결과를 보고 고르면 수치가
좋아 보이게 손잡이를 맞추는 것이고, 그것을 막기 위해
`PREREG_gripper_binary_head_0910.md` 가 있다."""


class ArmL1GripperBCE(nn.Module):
    """Regression loss on the arm channels, BCE on the gripper channel.
    팔 채널은 회귀 손실, 그리퍼 채널은 BCE.

    The arm term is left exactly as it was so only one thing changes at a time:
    if both the arm loss and the gripper loss moved, a difference in the rollout
    rate could not be attributed to either.
    팔 항은 기존과 정확히 같게 둔다. 한 번에 하나만 바꿔야 하기 때문이다 — 팔 손실과
    그리퍼 손실이 동시에 바뀌면 롤아웃 성공률 차이를 어느 쪽에도 귀속시킬 수 없다.

    `pos_weight` compensates the class imbalance: the gripper is open on roughly
    110 of 141 steps, so without it the majority class dominates the gradient in
    the same way the L1 median did.
    `pos_weight` 는 클래스 불균형을 보정한다. 그리퍼는 141스텝 중 약 110이 열림이라,
    없으면 L1 중앙값이 그랬던 것과 같은 방식으로 다수 클래스가 기울기를 지배한다.

    ⚠️ 손실 값은 기존 공간의 손실과 **비교할 수 없다.** 항이 두 개고 단위가 다르다.
       비교는 언제나 롤아웃 성공률로 한다.
    """

    def __init__(
        self,
        base: nn.Module,
        gripper: int,
        pos_weight: float,
        weight: float = GRIPPER_LOSS_WEIGHT,
    ) -> None:
        super().__init__()
        self.base = base
        self.gripper = int(gripper)
        self.pos_weight = float(pos_weight)
        self.weight = float(weight)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        arm = [i for i in range(pred.shape[-1]) if i != self.gripper]
        arm_loss = self.base(pred[..., arm], target[..., arm])
        grip_loss = nn.functional.binary_cross_entropy_with_logits(
            pred[..., self.gripper],
            target[..., self.gripper],
            # 텐서를 forward 에서 만든다. 버퍼로 두면 criterion 을 .to(device) 하지
            # 않는 현재 호출 경로에서 장치가 갈린다.
            pos_weight=target.new_tensor(self.pos_weight),
        )
        return arm_loss + self.weight * grip_loss


def make_loss(
    name: str,
    action_space: str | None = None,
    gripper_pos_weight: float | None = None,
) -> nn.Module:
    """L1 or MSE, wrapped in a gripper BCE term for the binary action space.
    L1 또는 MSE. 이진 행동공간이면 그리퍼 BCE 항으로 감싼다.

    Recorded in the checkpoint so a number can be attributed.
    수치를 귀속시킬 수 있도록 체크포인트에 기록한다."""
    if name == "l1":
        base: nn.Module = nn.L1Loss()
    elif name == "mse":
        base = nn.MSELoss()
    else:
        raise ValueError(f"loss 는 l1|mse 여야 한다: {name}")

    if action_space != "joint_delta_gripper_binary":
        return base
    if gripper_pos_weight is None:
        raise ValueError(
            "joint_delta_gripper_binary 는 gripper_pos_weight 가 필요하다. "
            "데이터셋의 열림/닫힘 비율에서 계산하며 손으로 정하지 않는다"
        )
    return ArmL1GripperBCE(base, gripper_index(), gripper_pos_weight)


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic train/val split.
    결정적인 train/val 분할."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    # val_fraction 0 means exactly that: no held-out samples. The memorisation
    # test needs the network to have seen every step of the episode it is then
    # asked to reproduce -- a forced single held-out sample would make one step
    # of the trace unreadable for no reason.
    # val_fraction 0 은 말 그대로 0 이다 — 홀드아웃 없음. 외우기 검사는 재현을 요구할
    # 에피소드의 모든 스텝을 신경망이 봤어야 하는데, 강제 홀드아웃 1개가 있으면
    # 추적의 한 스텝이 이유 없이 읽을 수 없게 된다.
    if val_fraction <= 0.0 or n <= 1:
        return idx.tolist(), []
    n_val = max(1, int(round(n * val_fraction)))
    return idx[n_val:].tolist(), idx[:n_val].tolist()


def split_by_episode(
    index: list[tuple[int, int]], n_episodes: int, val_fraction: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Hold out whole episodes, not individual ticks.
    틱이 아니라 에피소드 단위로 홀드아웃한다.

    A tick-level split puts frames from the same episode on both sides. The val
    frames then sit milliseconds away from training frames of the same object
    position and the same trajectory, so val loss measures interpolation within
    a seen episode -- not whether the policy handles an object it has never seen.
    Measured: val 0.043 alongside a 5.7% rollout. The episode split is the one
    that asks the rollout's question.
    틱 단위 분할은 같은 에피소드의 프레임을 양쪽에 놓는다. val 프레임은 같은 물체
    위치·같은 궤적의 학습 프레임에서 몇 ms 떨어져 있을 뿐이어서, val 손실은 본
    에피소드 안의 보간을 재는 것이고 처음 보는 물체를 다루는지를 재지 않는다.
    실측: val 0.043 인데 롤아웃 5.7%. 에피소드 분할이 롤아웃과 같은 질문을 던진다.

    Returns (train sample idx, val sample idx, val episode idx).
    """
    rng = np.random.default_rng(seed)
    eps = rng.permutation(n_episodes)
    n_val = 0 if val_fraction <= 0.0 or n_episodes <= 1 else max(1, int(round(n_episodes * val_fraction)))
    val_eps = set(int(e) for e in eps[:n_val])
    tr = [i for i, (e, _t) in enumerate(index) if e not in val_eps]
    va = [i for i, (e, _t) in enumerate(index) if e in val_eps]
    return tr, va, sorted(val_eps)


@dataclass
class TargetTransform:
    """The one place that decides what the loss is computed against.
    손실이 무엇을 대상으로 계산되는지를 정하는 유일한 자리.

    It exists because it was optional and got left out. `run_epoch` took
    `target_mean`/`target_std` as keyword defaults; the training call passed
    `optimizer` and `grad_clip` positionally and omitted them, so training
    regressed the raw delta while validation and inference assumed a
    standardised one. Nothing raised. The train loss was printed in one unit and
    compared against a baseline printed in another, and a policy that had learned
    nothing looked like a policy that had memorised its episode. 🟢 2026-09-02
    선택 인자였기 때문에 빠졌다. `run_epoch` 이 `target_mean`/`target_std` 를 기본값
    키워드로 받았고, 학습 호출이 `optimizer` 와 `grad_clip` 을 위치로 넘기면서 둘을
    누락했다. 그래서 학습은 raw 델타를 회귀하고 검증과 추론은 표준화된 것을 가정했다.
    아무 곳에서도 예외가 나지 않았다. 학습 손실은 한 단위로 출력되고 다른 단위의
    baseline 과 비교됐으며, 아무것도 배우지 못한 정책이 에피소드를 외운 정책처럼
    보였다.

    So it is now one required positional object instead of two optional keywords.
    그래서 선택 키워드 두 개가 아니라 **필수 위치 인자 하나**로 바꿨다.
    """

    action_space: str
    mean: torch.Tensor | None
    std: torch.Tensor | None

    def __call__(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Actions and states in, the tensor the loss sees out.
        행동·상태를 받아 손실이 보는 텐서를 낸다."""
        target = training_target(action, state, self.action_space)
        if self.mean is not None and self.std is not None:
            target = (target - self.mean) / self.std
        return target

    def to(self, device: torch.device) -> "TargetTransform":
        """Move the statistics to the training device.
        통계를 학습 디바이스로 옮긴다."""
        if self.mean is None or self.std is None:
            return self
        return TargetTransform(self.action_space, self.mean.to(device), self.std.to(device))

    @property
    def standardised(self) -> bool:
        """Whether the loss is computed in standardised units.
        손실이 표준화 단위로 계산되는가."""
        return self.mean is not None and self.std is not None


def run_epoch(
    model: BCNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_fn: TargetTransform,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
) -> float:
    """One pass. `optimizer=None` means evaluation.
    한 바퀴. `optimizer=None` 이면 평가.

    ⚠️ 손실은 `model.action_space` 가 정하는 공간에서 계산된다. 절대 목표의 손실과
       델타 목표의 손실은 **서로 비교할 수 없다** — 크기가 두 자릿수 다르다.
       비교는 언제나 롤아웃 성공률로 한다.
    """
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for images, state, action in loader:
            images = {c: v.to(device) for c, v in images.items()}
            state, action = state.to(device), action.to(device)
            pred = model(images, state)
            loss = criterion(pred, target_fn(action, state))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += float(loss.item()) * action.shape[0]
            n += action.shape[0]
    return total / max(n, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="BC 정책 학습")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="계약 에피소드 디렉터리")
    src.add_argument("--random", type=int, metavar="N",
                     help="랜덤 텐서 N개로 루프만 검증한다 (데이터 불필요)")
    parser.add_argument("--epochs", type=int, default=None, help="설정값을 덮어쓴다")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split-by", choices=("sample", "episode"), default=None,
                        help="val 분할 단위. 설정값(train.val_split)을 덮어쓴다. "
                             "episode 면 에피소드 통째로 홀드아웃 — 일반화를 재는 유일한 분할")
    parser.add_argument("--val-fraction", type=float, default=None,
                        help="설정값을 덮어쓴다. 0 이면 홀드아웃 없음 (외우기 검사용)")
    parser.add_argument("--image-noise", type=float, default=None,
                        metavar="GRAY",
                        help="학습 이미지에 더할 가우시안 잡음의 표준편차, 단위는 계조(0~255). "
                             "설정값을 덮어쓴다. val 은 흔들지 않는다")
    parser.add_argument("--seed", type=int, default=None,
                        help="설정값을 덮어쓴다. 학습 3회 반복 시 서로 다른 값을 준다")
    parser.add_argument("--out", type=Path, default=None, help="체크포인트 경로")
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument(
        "--action-space", type=str, default=None,
        help="configs/train/bc.yaml 의 model.action_space 를 덮어쓴다. "
             "설정 파일을 고치지 않고 조건을 바꾸기 위한 것이며 "
             "EXP_LOG conditions.action_space 에 실제 사용값이 기록된다. "
             "⚠️ train_config_sha 는 파일 해시라 이 덮어쓰기를 반영하지 않는다",
    )
    parser.add_argument(
        "--cameras", type=str, default=None,
        help="쉼표로 구분한 카메라 이름. 데이터셋 카메라의 **부분집합**만 쓴다. "
             "예: --cameras cam_wrist (실물 배포 구성. L76)",
    )
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg: dict[str, Any] = load_train_config()
    if args.action_space is not None:
        cfg["model"]["action_space"] = args.action_space
        print(f"· 행동공간 덮어쓰기: {args.action_space} (설정 파일은 건드리지 않았다)")
    t = cfg["train"]
    epochs = int(args.epochs if args.epochs is not None else t["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else t["batch_size"])
    seed = int(args.seed if args.seed is not None else t["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)

    if args.random is not None:
        cameras = (
            [c.strip() for c in args.cameras.split(",") if c.strip()]
            if args.cameras else DEFAULT_CAMERAS
        )
        dataset: Any = RandomTensorDataset(args.random, cfg, cameras, seed=seed)
        trained_on = "random_tensors"
        n_episodes = 0
        print("⚠️ 랜덤 텐서로 학습한다. **손실 값에 의미가 없다.**")
        print("   확인하는 것은 루프가 끝까지 도는가 하나뿐이다.\n")
    else:
        cam_override = (
            [c.strip() for c in args.cameras.split(",") if c.strip()]
            if args.cameras else None
        )
        dataset = EpisodeDataset(args.data, cfg, camera_names=cam_override)
        cameras = dataset.camera_names
        if cam_override is not None:
            print(f"· 카메라 덮어쓰기: {cameras} (데이터셋은 건드리지 않았다)")
        trained_on = str(args.data)
        n_episodes = len(dataset.episodes)

    print(dataset.summary())

    val_fraction = float(args.val_fraction if args.val_fraction is not None
                         else t["val_fraction"])
    split_by = str(args.split_by if args.split_by is not None else t.get("val_split", "sample"))
    val_episodes: list[int] = []
    if split_by == "episode" and getattr(dataset, "index", None) is not None:
        tr_idx, va_idx, val_episodes = split_by_episode(
            dataset.index, len(dataset.episodes), val_fraction, seed
        )
        print(f"분할: **episode 단위** · val 에피소드 {len(val_episodes)}개 → "
              f"train {len(tr_idx)} / val {len(va_idx)} 샘플 (val_fraction {val_fraction})")
        print("  val 은 학습에 없는 물체 위치다 — 이 val 은 일반화를 잰다")
    else:
        tr_idx, va_idx = split_indices(len(dataset), val_fraction, seed)
        print(f"분할: sample 단위 · train {len(tr_idx)} / val {len(va_idx)} (val_fraction {val_fraction})"
              + ("  ⚠️ 홀드아웃 없음 — 외우기 검사 전용. 일반화 수치가 아니다" if not va_idx else
                 "  ⚠️ 같은 에피소드의 다른 틱이 val 이다 — 일반화를 재지 않는다"))
    # 증강은 **분할이 확정된 뒤에** 건다. train 인덱스에만 걸어야 val 이 깨끗하다.
    noise_gray = float(args.image_noise if args.image_noise is not None
                       else t.get("image_noise_gray", 0.0))
    if noise_gray > 0.0 and hasattr(dataset, "set_image_noise"):
        dataset.set_image_noise(noise_gray, tr_idx)
        print(f"이미지 잡음: 학습 입력에 σ={noise_gray:.1f} 계조 (val 은 원본)")
        print("  근거 🟢 폐루프에서 관측이 t=1 에 0.28 계조, t=10 에 1.19 계조 갈라진다. "
              "L39: 1 계조가 성공/실패를 뒤집는다")
    elif noise_gray > 0.0:
        print(f"⚠️ 이미지 잡음 {noise_gray} 계조를 요청했으나 이 데이터셋은 지원하지 않는다 — 무시한다")

    loaders = {
        "train": DataLoader(Subset(dataset, tr_idx), batch_size=batch_size, shuffle=True,
                            num_workers=int(t["num_workers"]), collate_fn=collate),
        "val": DataLoader(Subset(dataset, va_idx), batch_size=batch_size, shuffle=False,
                          num_workers=int(t["num_workers"]), collate_fn=collate)
        if va_idx else None,
    }

    # Target statistics, computed from the episode arrays rather than by iterating
    # the DataLoader -- the loader would decode 13,818 images to read six numbers.
    # 타겟 통계. DataLoader 를 도는 대신 에피소드 배열에서 직접 계산한다. 로더로 돌면
    # 숫자 여섯 개를 읽으려고 이미지 13,818장을 디코딩한다.
    t_mean = t_std = None
    baseline_norm = float("nan")
    grip_pos_weight: float | None = None
    trivial_label = "항상 0"
    # Defined before the branch so the random-tensor path cannot reach the epoch
    # loop without one. An undefined transform there would be a NameError at the
    # first epoch, i.e. a crash instead of a silent unit mismatch -- but the point
    # of this object is that neither is possible.
    # 랜덤 텐서 경로가 이것 없이 epoch 루프에 도달할 수 없도록 분기 앞에서 정의한다.
    target_fn = TargetTransform(
        str(cfg["model"].get("action_space", "joint_absolute")), None, None
    )
    if getattr(dataset, "episodes", None):
        acts = torch.from_numpy(
            np.concatenate([e.action for e in dataset.episodes], axis=0)
        ).float()
        sts = torch.from_numpy(
            np.concatenate([e.state for e in dataset.episodes], axis=0)
        ).float()
        space = str(cfg["model"].get("action_space", "joint_absolute"))
        raw_target = training_target(acts, sts, space)
        if bool(t.get("normalize_target", True)):
            t_mean, t_std = target_scale(raw_target, space)
        # Built here and used by BOTH passes. The baseline below is computed
        # through the same object, so the epoch losses and the reference they are
        # read against cannot end up in different units.
        # 여기서 만들어 **양쪽 패스**가 쓴다. 아래 baseline 도 같은 객체를 통과하므로,
        # epoch 손실과 그것을 읽는 기준이 다른 단위가 될 수 없다.
        target_fn = TargetTransform(space, t_mean, t_std)
        scaled = target_fn(acts, sts)
        # What the head is asked to fit, joint by joint, before any scaling. A
        # target that is exactly zero on most steps (the gripper delta was zero
        # on 124 of 141) has median zero, and an L1 head learns that median.
        # Seen here before training instead of after a 0% rollout.
        # 스케일링 전, 관절별로 헤드가 맞춰야 하는 것. 대부분 스텝에서 정확히 0 인
        # 목표(그리퍼 델타는 141 중 124 가 0 이었다)는 중앙값이 0 이고 L1 헤드는 그
        # 중앙값을 배운다. 0% 롤아웃 뒤가 아니라 학습 전에 여기서 본다.
        zero_frac = (raw_target.abs() < 1e-6).float().mean(dim=0)
        print("회귀 목표 분포 (스케일링 전): 관절별 |std| / 정확히 0 인 비율")
        print("  std  " + " ".join(f"{float(v):.5f}" for v in raw_target.std(dim=0)))
        print("  zero " + " ".join(f"{float(v):6.0%}" for v in zero_frac))
        if bool((zero_frac > 0.5).any()):
            print("  ⚠️ 절반 넘게 0 인 관절이 있다 — L1 의 최적값이 0 이 되는 스파이크 목표다")
        # What a network that always outputs zero would score, in the same units
        # the epoch losses are printed in. A loss without this reference cannot be
        # read: 0.0057 looked fine until it turned out zero-output scores 0.0075.
        # 항상 0 을 내는 신경망의 점수. epoch 손실과 같은 단위다. 이 기준 없이는 손실을
        # 읽을 수 없다 — 0.0057 이 괜찮아 보였는데 0 출력이 0.0075 였다.
        baseline_norm = float(scaled.abs().mean())
        if space == "joint_delta_gripper_binary":
            g = gripper_index()
            closed = float(raw_target[:, g].sum())
            opened = float(raw_target.shape[0]) - closed
            if closed < 1.0:
                raise ValueError(
                    "이진 라벨에 닫힘 프레임이 없다. "
                    "grasp.close_cmd/open_cmd 와 데이터의 그리퍼 명령을 대조하라"
                )
            # 데이터에서 파생되는 통계다. 손으로 고르는 하이퍼파라미터가 아니다.
            grip_pos_weight = opened / closed
            print(f"그리퍼 이진 라벨: 닫힘 {closed:.0f} / 열림 {opened:.0f} "
                  f"({closed / raw_target.shape[0]:.1%}) · "
                  f"pos_weight {grip_pos_weight:.3f} (데이터에서 계산)")
            # 자명한 예측기를 "항상 0" 이 아니라 **항상 열어둠** 으로 바꾼다.
            # 이 공간에서 이기고 싶은 상대가 그것이다 — v5 는 100편 중 42편에서
            # 그리퍼를 한 번도 닫지 않았다 🟢 2026-09-10.
            ref = make_loss(str(t["loss"]), space, grip_pos_weight)
            trivial = torch.zeros_like(scaled)
            trivial[:, g] = -4.0
            baseline_norm = float(ref(trivial, scaled))
            trivial_label = "항상 열어둠"

    model = BCNet(cameras, cfg).to(device)
    target_fn = target_fn.to(device)
    if t_mean is not None:
        t_mean, t_std = t_mean.to(device), t_std.to(device)
    if model.action_space != target_fn.action_space:
        raise ValueError(
            f"손실 목표 공간({target_fn.action_space})과 모델 행동 공간"
            f"({model.action_space})이 다르다. 체크포인트를 불러올 때 어긋난다"
        )
    n_params = model.n_params()

    # The action space is a one-line config change that silently decides what the
    # loss even means. If it does not take effect, training runs for hours in the
    # wrong space and the loss looks fine. So it gets printed, not assumed.
    # 행동 공간은 설정 한 줄이지만 손실의 의미 자체를 정한다. 이게 안 먹으면 몇 시간을
    # 엉뚱한 공간에서 학습하고도 손실은 멀쩡해 보인다. 그래서 가정하지 않고 출력한다.
    print(f"행동 공간: {model.action_space}", end="")
    if model.action_space == "joint_delta_gripper_abs":
        print("  — 팔 관절은 action - state 잔차, 그리퍼는 절대 명령")
        print("  ⚠️ 손실 값을 절대 목표 학습분과 비교하지 마라. 크기가 두 자릿수 다르다")
    elif model.action_space == "joint_delta":
        print("  — 목표는 action - state 잔차. 출력에 state 를 더해 행동을 만든다")
        print("  ⚠️ 손실 값을 절대 목표 학습분과 비교하지 마라. 크기가 두 자릿수 다르다")
    else:
        print("  — 목표는 절대 관절각")
        print("  ⚠️ 실측상 이 공간에서는 상태만으로 정답의 86% 가 설명된다"
              " (2026-09-02 image_sensitivity)")
    if t_std is not None:
        print(f"타겟 표준화: 켬 (관절별 std {[round(float(v), 5) for v in t_std.cpu()]})")
    else:
        print("타겟 표준화: 끔")
    unit = "표준화 단위" if target_fn.standardised else "raw 델타 단위"
    print(f"손실 단위: {unit} — train·val·아래 baseline 이 모두 이 단위다")
    if baseline_norm == baseline_norm:  # not NaN
        print(f"⚠️ 자명한 예측기({trivial_label}) 손실 = {baseline_norm:.5f}")
        print("   epoch 손실이 이 값 근처에서 멈추면 학습이 안 되고 있는 것이다."
              " 손실이 내려간 것만으로 판단하지 마라")
    print(f"파라미터 {n_params:,}개 (~{n_params * 4 / 1024 / 1024:.1f}MB fp32)")
    print("⚠️ Jetson 8GB 에 VLM 과 함께 올라가야 한다 — 이슈 42 미검증\n")

    criterion = make_loss(
        str(t["loss"]), target_fn.action_space, grip_pos_weight
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]),
                                  weight_decay=float(t["weight_decay"]))

    out = args.out or (AI_ROOT / cfg["checkpoint"]["dir"] /
                       ("bc_random.pt" if args.random else "bc.pt"))
    best_out = out.with_name(out.stem + "_bestval" + out.suffix)
    best = float("inf")
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        tr = run_epoch(model, loaders["train"], criterion, device, target_fn,
                       optimizer, float(t["grad_clip"]))
        va = (run_epoch(model, loaders["val"], criterion, device, target_fn)
              if loaders["val"] is not None else float("nan"))
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        mark = ""
        if loaders["val"] is not None and va < best:
            best = va
            save_checkpoint(best_out, model, CheckpointMeta(
                camera_names=list(cameras), contract_version=CONTRACT_VERSION,
                action_space=model.action_space,
                target_mean=[float(v) for v in t_mean.cpu()] if t_mean is not None else None,
                target_std=[float(v) for v in t_std.cpu()] if t_std is not None else None,
                train_config=cfg, config_sha=file_digest(DEFAULT_CONFIG),
                code_sha=code_digest(), n_params=n_params, n_episodes=n_episodes,
                n_samples=len(dataset), epochs_run=ep, best_val_loss=best,
                trained_on=trained_on))
            mark = "  ← best-val 저장"
        print(f"  epoch {ep:3d}/{epochs}  train {tr:.5f}  val {va:.5f}{mark}")

    elapsed = time.perf_counter() - t0

    # The main checkpoint is the LAST epoch, not the one with the lowest val loss.
    # This project's own rule says val loss only tells you whether training broke,
    # and yet checkpoint selection was being driven by it -- on a flat val curve
    # that picks an arbitrary early epoch. Measured: a 200-epoch run saved its
    # epoch-1 model, and the rollout that "evaluated the policy" evaluated an
    # untrained network. 🟢 2026-09-02
    # 주 체크포인트는 val loss 가 가장 낮은 epoch 이 아니라 **마지막 epoch** 이다.
    # 이 프로젝트 규칙은 val loss 가 "학습이 망가졌나"만 말한다고 해놓고, 정작
    # 체크포인트 선택을 그것이 하고 있었다. val 곡선이 평평하면 임의의 이른 epoch 이
    # 뽑힌다. 실측: 200 epoch 실행이 epoch 1 모델을 저장했고, "정책을 평가"한 롤아웃이
    # 학습되지 않은 신경망을 평가했다.
    save_checkpoint(out, model, CheckpointMeta(
            camera_names=list(cameras), contract_version=CONTRACT_VERSION,
            action_space=model.action_space,
            target_mean=[float(v) for v in t_mean.cpu()] if t_mean is not None else None,
            target_std=[float(v) for v in t_std.cpu()] if t_std is not None else None,
            train_config=cfg, config_sha=file_digest(DEFAULT_CONFIG),
            code_sha=code_digest(), n_params=n_params, n_episodes=n_episodes,
            n_samples=len(dataset), epochs_run=epochs, best_val_loss=best,
            trained_on=trained_on))

    print(f"\n체크포인트(마지막 epoch): {out}  ({elapsed:.1f}초, {elapsed / epochs:.2f}초/epoch)")
    if best_out.exists():
        print(f"참고용(best val): {best_out}")
        print("  ⚠️ 판정에는 마지막 epoch 을 쓴다. val loss 가 낮은 epoch 이 좋은 정책이라는"
              " 근거가 이 프로젝트에는 없다")
    print("\n" + "=" * 70)
    print("⚠️ 손실은 성과가 아니다. 이 체크포인트가 쓸 만한지는 아직 모른다.")
    print("   판정은 롤아웃 성공률이고, 넘어야 할 값은 20.0% 다:")
    print(f"     python tools/eval_rollout.py --policy-ckpt {out} --render --log")
    if args.random is not None:
        print("   ⚠️ 지금 것은 랜덤 텐서 학습이다. 평가해도 의미 없다.")
    print("=" * 70)

    if args.log:
        rec = log_run(
            experiment="train_bc", author=args.author, issue="S15P21A103-34",
            conditions={
                "trained_on": trained_on, "n_episodes": n_episodes,
                "n_samples": len(dataset), "epochs": epochs, "batch_size": batch_size,
                "split_by": split_by, "val_fraction": val_fraction,
                "val_episodes": val_episodes,
                "lr": t["lr"], "loss": t["loss"], "seed": seed, "device": str(device),
                "encoder_mode": cfg["model"]["encoder_mode"],
                "action_space": model.action_space,
                "normalize_target": t_std is not None,
                "trivial_baseline_loss": baseline_norm, "n_params": n_params,
                "trivial_baseline_kind": trivial_label,
                "gripper_pos_weight": grip_pos_weight,
                "gripper_loss_weight": (
                    GRIPPER_LOSS_WEIGHT
                    if model.action_space == "joint_delta_gripper_binary"
                    else None
                ),
                "config_sha": file_digest(DEFAULT_CONFIG),
                "metric_note": "손실은 학습이 망가졌는지 확인용. 판정은 롤아웃 성공률(게이트 20.0%)",
            },
            result={
                "best_val_loss": best if best != float("inf") else None,
                "final_train_loss": history[-1]["train_loss"] if history else None,
                "history": history, "seconds": round(elapsed, 2),
                "checkpoint": str(out),
                "success_rate": None,
                "success_rate_note": "여기서 재지 않는다. tools/eval_rollout.py 참조",
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (code {rec['code_sha']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
