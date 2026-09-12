"""Behavior cloning — the baseline learned policy. Deliberately the simplest one.
행동복제 — 기준이 되는 학습 정책. 의도적으로 가장 단순한 것.

BC before ACT before Diffusion. If BC's number is not measured first, a later
model that works cannot be explained and one that fails cannot be diagnosed —
you will not know whether the problem is the data or the model.
BC 다음 ACT 다음 Diffusion. BC 수치를 먼저 재지 않으면, 나중 모델이 잘 돼도
이유를 설명 못 하고 안 돼도 원인(데이터 vs 모델)을 좁힐 수 없다.

⚠️ 통과 기준은 validation loss 가 아니다. 롤아웃 성공률이고, 넘어야 할 값은
   **20.0%** 다 (eval/rollout.py 의 GATES, 결과 보기 전에 확정됨).
   손실은 "학습이 망가지지 않았나" 확인용으로만 쓴다.

⚠️ Jetson 8GB 에 VLM 과 함께 올라가야 한다. 파라미터 수를 체크포인트 메타에
   기록하고, 학습 시작 때 출력한다. 키우기 전에 이슈 42 를 확인한다.
"""

from __future__ import annotations

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
import yaml

from paths import CONFIG_DIR, DEFAULT_CONFIG
from policy.base import check_action
from sim.base import Observation

DEFAULT_TRAIN_CONFIG = CONFIG_DIR / "train" / "bc.yaml"


def load_train_config(path: Path = DEFAULT_TRAIN_CONFIG) -> dict[str, Any]:
    """Read the BC training config.
    BC 학습 설정을 읽는다."""
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class ConvEncoder(nn.Module):
    """A small strided CNN. Not pretrained, on purpose — for now.
    작은 스트라이드 CNN. 사전학습 없음, 지금은 의도적으로.

    An ImageNet-pretrained ResNet would very likely help with a small dataset,
    and it is the obvious next thing to measure. It is not here yet because the
    point of this first pass is a loop that runs end to end without a download,
    and because swapping the encoder changes the image normalization statistics
    — which is a change worth recording rather than sliding in.
    작은 데이터셋에서는 ImageNet 사전학습 ResNet 이 도움이 될 가능성이 높고,
    다음에 재볼 후보다. 지금 없는 이유는 이번 단계의 목적이 다운로드 없이
    끝까지 도는 루프이고, 인코더를 바꾸면 이미지 정규화 통계가 함께 바뀌기
    때문이다 — 슬쩍 넣을 것이 아니라 기록할 변경이다.
    """

    def __init__(self, channels: list[int], feature_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float -> (B, feature_dim).
        (B, 3, H, W) float -> (B, feature_dim)."""
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)


ACTION_SPACES = (
    "joint_absolute",
    "joint_delta",
    "joint_delta_gripper_abs",
    "joint_delta_gripper_binary",
)
"""What the network regresses. Not a contract change -- the dataset always stores
absolute joint angles and this is a transform applied at train and inference time.
신경망이 무엇을 회귀하는가. 계약 변경이 아니다. 데이터셋에는 언제나 절대 관절각이
저장되고, 이것은 학습·추론 시점에 적용되는 변환이다.

joint_delta_gripper_abs: arm joints as `action - state`, gripper as the absolute
command. The gripper command is a set-point that changes twice per episode (open,
then close), so its delta is zero on 124 of 141 steps and a spike on the other 17.
Under an L1 loss the optimum for a mostly-zero target is its median -- zero -- and
the network learned exactly that: arm-joint error fell to 0.17x the required motion
while the gripper stayed at 1.07x, and in closed loop the gripper never opened.
🟢 2026-09-02 trace_execution, one-episode memorisation test.
joint_delta_gripper_abs: 팔 관절은 `action - state`, 그리퍼는 절대 명령. 그리퍼
명령은 에피소드당 두 번(열기, 닫기) 바뀌는 설정값이라 델타가 141스텝 중 124스텝에서
0 이고 17스텝에서만 스파이크다. L1 손실에서 대부분 0 인 목표의 최적값은 중앙값 — 0 —
이고 신경망은 정확히 그것을 배웠다. 팔 관절 오차는 필요한 움직임의 0.17배까지
내려갔는데 그리퍼는 1.07배에 머물렀고, 폐루프에서 그리퍼가 열리지 않았다.

joint_delta_gripper_binary: arm joints as `action - state`, gripper as a binary
label (1 = closed) fitted with BCE instead of the regression loss. Making the
gripper absolute fixed the zero-delta collapse above, but the same structure came
back one level up: the absolute command is "open" on roughly 110 of 141 steps, so
under L1 the conditional median is "open" wherever the observation cannot resolve
the closing moment -- and the policy simply never closes. Measured 🟢 2026-09-10:
v5 seed0 never closed the gripper in 42 of 100 episodes (scripted: 1 of 100), and
where it did close it closed at a median 28.0mm from the object (scripted 4.7mm).
BCE has no median-collapse: it fits a graded probability, so a weakly informative
observation still produces a crossing.
joint_delta_gripper_binary: 팔 관절은 `action - state`, 그리퍼는 **이진 라벨**
(1 = 닫힘)이고 회귀 손실 대신 BCE 로 맞춘다. 그리퍼를 절대값으로 바꾼 것이 위의
델타-0 붕괴는 고쳤지만, **같은 구조가 한 단계 위에서 재발했다** — 절대 명령은 141
스텝 중 약 110 스텝이 "열림"이라, 관측이 닫는 순간을 분해하지 못하는 구간에서 L1 의
조건부 중앙값은 "열림"이고 정책은 아예 닫지 않는다. 실측 🟢 2026-09-10: v5 seed0 이
100편 중 **42편에서 그리퍼를 한 번도 닫지 않았다** (scripted 는 1편), 닫은 경우도
물체에서 중앙 28.0mm 떨어진 곳에서 닫았다 (scripted 4.7mm). BCE 는 중앙값 붕괴가
없다 — 등급이 있는 확률을 맞추므로 관측이 약하게만 정보를 줘도 통과가 생긴다."""


def gripper_index(config_path: Path = DEFAULT_CONFIG) -> int:
    """Position of the gripper in the state/action vector, read from the hardware config.
    state/action 벡터에서 그리퍼의 위치. 하드웨어 설정에서 읽는다.

    Hardware-dependent, so it lives in configs/so101.yaml and not here.
    하드웨어 의존 값이므로 여기가 아니라 configs/so101.yaml 에 있다.
    """
    with Path(config_path).open(encoding="utf-8") as fh:
        joints = yaml.safe_load(fh)["joints"]
    for j in joints:
        if j["name"] == "gripper":
            return int(j["index"])
    raise KeyError(f"{config_path} 의 joints 에 'gripper' 가 없다")


_GRIPPER = gripper_index()


def gripper_command_norms(
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[float, float]:
    """Normalised open and close gripper commands, from the hardware config.
    정규화된 그리퍼 열기·닫기 명령. 하드웨어 설정에서 읽는다.

    Hardware-dependent and contract-defined, so nothing here is a literal: the
    commands come from `grasp.open_cmd`/`grasp.close_cmd` and the mapping is the
    contract's own formula. The arm is going to be replaced, so a number typed
    here would go stale silently.
    하드웨어 의존 + 계약 정의 값이므로 리터럴을 쓰지 않는다. 명령은
    `grasp.open_cmd`/`grasp.close_cmd` 에서, 매핑은 계약 자신의 공식에서 온다.
    로봇팔은 교체 예정이라 여기 숫자를 박으면 조용히 낡는다.
    """
    with Path(config_path).open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    lo, hi = (float(v) for v in cfg["joints"][_GRIPPER]["range_rad"])
    grasp = cfg["grasp"]

    def unit(x: float) -> float:
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    return unit(float(grasp["open_cmd"])), unit(float(grasp["close_cmd"]))


_GRIP_OPEN_NORM, _GRIP_CLOSE_NORM = gripper_command_norms()
_GRIP_MID = (_GRIP_OPEN_NORM + _GRIP_CLOSE_NORM) / 2.0

# 데이터에서 두 모드를 찾을 때 쓰는 분위수. 양 끝의 이상치를 피하면서 두 평탄부를 집는다.
GRIP_QUANTILES = (0.05, 0.95)
# 두 모드가 전체 범위의 이 비율보다 가깝게 붙어 있으면 개폐 두 모드가 없는 것이다.
GRIP_MIN_SEPARATION = 0.05
# 중간 띠(두 모드 사이 10% 구간)에 이 비율보다 많이 들어 있으면 이봉분포가 아니다.
GRIP_MAX_MIDBAND_FRAC = 0.15


def gripper_norms_from_data(
    gripper_channel: torch.Tensor,
) -> tuple[float, float, float]:
    """Open / close / threshold read from the data, not from the sim config.
    열림·닫힘·임계값을 시뮬 설정이 아니라 **데이터에서** 읽는다.

    Why this exists / 왜 있는가 (2026-09-12) 🟢.

    `_GRIP_MID` 는 `grasp.open_cmd`/`close_cmd` 에서 파생된다. 그 둘은 **시뮬** 값이다 --
    2cm 큐브를 1.8cm 로 무는 설정이라 정규화하면 열림 -0.1931 · 닫힘 -0.7557 · 중간 -0.4744 다.
    실물 UMI 시연은 6.7cm 로 벌려 4cm 상자를 물어서 열림 +0.2964 · 닫힘 -0.3414 이고,
    **둘 다 -0.4744 보다 위**다. 그 임계값을 그대로 쓰면 실물 프레임의 닫힘 라벨이 **0/76** 이 된다.

    같은 채널에 규약이 두 개인 것이고 L69(action 의미 두 갈래)와 같은 계열이다.
    임계값은 하드웨어 설정이 아니라 **그 데이터셋이 쓰는 규약**에서 나와야 한다.

    검증 가능한 성질: 시뮬 데이터에 적용하면 설정에서 파생된 값이 그대로 나온다
    (`tools/check_gripper_norms.py` 가 그것을 fixture 로 검정한다).

    Returns (open_norm, close_norm, mid). 닫힘이 열림보다 작다 -- 계약상 range_rad
    하한이 닫힘이다.
    """
    x = gripper_channel.reshape(-1).to(torch.float64)
    lo_q, hi_q = GRIP_QUANTILES
    close_norm = float(torch.quantile(x, lo_q))
    open_norm = float(torch.quantile(x, hi_q))
    span = float(x.max() - x.min())

    if span <= 0.0 or (open_norm - close_norm) < GRIP_MIN_SEPARATION * max(span, 1e-9):
        raise ValueError(
            "그리퍼 채널에 개폐 두 모드가 없다 "
            f"(q{lo_q:.2f}={close_norm:.4f} · q{hi_q:.2f}={open_norm:.4f} · 전체폭 {span:.4f}). "
            "이 데이터로는 이진 라벨을 만들 수 없다 -- 폐쇄 이벤트가 있는지 먼저 확인하라"
        )

    mid = (open_norm + close_norm) / 2.0
    band = 0.10 * (open_norm - close_norm)
    midband = float(((x > mid - band) & (x < mid + band)).to(torch.float64).mean())
    if midband > GRIP_MAX_MIDBAND_FRAC:
        raise ValueError(
            f"그리퍼 채널이 이봉분포가 아니다 -- 중간 띠에 {midband:.1%} 가 있다 "
            f"(허용 {GRIP_MAX_MIDBAND_FRAC:.0%}). 이진화 임계값이 임의값이 된다"
        )
    return open_norm, close_norm, mid


def training_target(
    action: torch.Tensor,
    state: torch.Tensor,
    action_space: str,
    grip_mid: float | None = None,
) -> torch.Tensor:
    """The tensor the loss is computed against.
    손실을 계산할 대상 텐서.

    With an absolute target most of the loss is spent on "where is the arm now",
    which the state input already answers -- measured at 86% on this dataset. The
    part that actually produces motion, `action - state`, is small and gets left
    wrong. Subtracting the state removes exactly the term the state explains, so
    what is left is the part that requires knowing where the object is.
    절대 목표에서는 손실 대부분이 "팔이 지금 어디 있나"에 쓰이는데, 그건 상태 입력이
    이미 답하고 있다 — 이 데이터셋에서 86% 로 실측됐다. 실제로 움직임을 만드는
    `action - state` 는 작고 틀린 채 남는다. 상태를 빼면 상태가 설명하던 항이 정확히
    사라지고, 남는 것은 물체가 어디 있는지 알아야 설명되는 몫이다.
    """
    if action_space == "joint_delta":
        return action - state
    if action_space == "joint_delta_gripper_abs":
        target = action - state
        target = target.clone()
        target[..., _GRIPPER] = action[..., _GRIPPER]
        return target
    if action_space == "joint_delta_gripper_binary":
        target = action - state
        target = target.clone()
        # 닫힘 명령이 열림 명령보다 작다 (range_rad 하한 = 닫힘). 중간점 기준으로
        # 이진화한다 — 여기 리터럴은 없다.
        # `grip_mid` 를 주면 **그 데이터셋에서 파생된** 임계값을 쓴다. 안 주면 시뮬
        # 설정에서 파생된 값이다 — 실물 데이터에는 그것이 맞지 않는다 (2026-09-12 🟢,
        # `gripper_norms_from_data` 의 주석 참조).
        mid = _GRIP_MID if grip_mid is None else float(grip_mid)
        target[..., _GRIPPER] = (
            action[..., _GRIPPER] < mid
        ).to(action.dtype)
        return target
    if action_space == "joint_absolute":
        return action
    raise ValueError(f"action_space 는 {ACTION_SPACES} 중 하나여야 한다: {action_space!r}")


def target_scale(
    target: torch.Tensor,
    action_space: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-joint mean and std of the regression target.
    회귀 목표의 관절별 평균과 표준편차.

    Needed because the contract normalises against the joint *range*, and a delta
    is a tiny fraction of that range -- roughly 0.0008 to 0.03. A final linear
    layer initialises to outputs of order 0.1, so most of training is spent
    shrinking toward zero and the structure that matters never gets fit. Measured:
    a network with 1.3M parameters failed to memorise a single 141-sample episode,
    ending only 24% better than predicting zero. 🟢 2026-09-02
    계약이 관절 **범위** 기준으로 정규화하는데 델타는 그 범위의 0.0008~0.03 밖에 안
    되기 때문에 필요하다. 마지막 선형층은 O(0.1) 규모 출력으로 초기화되므로 학습
    대부분이 0 쪽으로 줄이는 데 쓰이고 정작 중요한 구조는 학습되지 않는다. 실측:
    1.3M 파라미터가 141샘플짜리 에피소드 하나를 외우지 못했고, "0 출력"보다 24%
    나은 데서 멈췄다.
    """
    mean = target.mean(dim=0)
    std = target.std(dim=0)
    # A joint that never moves has std 0. Dividing by it would produce inf, and a
    # constant target needs no scaling anyway.
    # 한 번도 안 움직인 관절은 std 가 0 이다. 그걸로 나누면 inf 가 되고, 상수 목표는
    # 애초에 스케일이 필요 없다.
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    if action_space == "joint_delta_gripper_binary":
        # 그리퍼 채널은 0/1 라벨이고 헤드 출력은 로짓이다. 표준화하면 BCE 가 보는
        # 라벨이 0/1 이 아니게 되고 sigmoid 임계값이 뜻을 잃는다. 항등으로 둔다.
        mean = mean.clone()
        std = std.clone()
        mean[_GRIPPER] = 0.0
        std[_GRIPPER] = 1.0
    return mean, std


def to_action(
    raw: torch.Tensor,
    state: torch.Tensor,
    action_space: str,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    grip_norms: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Turn the network's output into a contract-unit action.
    신경망 출력을 계약 단위 행동으로 바꾼다."""
    if target_mean is not None and target_std is not None:
        raw = raw * target_std + target_mean
    if action_space == "joint_delta":
        return state + raw
    if action_space == "joint_delta_gripper_abs":
        out = state + raw
        out = out.clone()
        out[..., _GRIPPER] = raw[..., _GRIPPER]
        return out
    if action_space == "joint_delta_gripper_binary":
        out = state + raw
        out = out.clone()
        # 로짓 > 0 이면 닫는다 (sigmoid > 0.5 와 같다). 중간값이 나올 수 없으므로
        # "명령 진폭이 문턱 미만" 으로 미폐쇄가 되는 경로가 구조적으로 사라진다.
        closed = raw[..., _GRIPPER] > 0.0
        # 내는 명령도 그 데이터셋의 규약이어야 한다. 실물 데이터로 학습한 정책이
        # 시뮬 명령(-0.7557 = 1.8cm)을 내면 4cm 물체를 뭉갠다.
        o_norm, c_norm = (_GRIP_OPEN_NORM, _GRIP_CLOSE_NORM) if grip_norms is None else grip_norms
        out[..., _GRIPPER] = torch.where(
            closed,
            torch.full_like(raw[..., _GRIPPER], c_norm),
            torch.full_like(raw[..., _GRIPPER], o_norm),
        )
        return out
    if action_space == "joint_absolute":
        return raw
    raise ValueError(f"action_space 는 {ACTION_SPACES} 중 하나여야 한다: {action_space!r}")


class BCNet(nn.Module):
    """Images (+ joint state) -> one action. Single step, no chunking.
    이미지 (+ 관절 state) -> 행동 하나. 단일 스텝, 청킹 없음.

    Action chunking is what ACT adds. Doing it here would mean BC and ACT differ
    in two ways at once, and a difference in the number could not be attributed.
    행동 청킹은 ACT 가 더하는 것이다. 여기서 하면 BC 와 ACT 가 두 가지가 동시에
    달라져서, 수치 차이를 무엇 때문인지 귀속시킬 수 없게 된다.
    """

    def __init__(
        self,
        camera_names: list[str],
        cfg: dict[str, Any],
        state_dim: int = 6,
        action_dim: int = 6,
    ) -> None:
        super().__init__()
        m = cfg["model"]
        self.camera_names = list(camera_names)
        self.encoder_mode = m["encoder_mode"]
        self.use_state = bool(m["use_state"])
        self.action_space = str(m.get("action_space", "joint_absolute"))
        if self.action_space not in ACTION_SPACES:
            raise ValueError(
                f"action_space 는 {ACTION_SPACES} 중 하나여야 한다: {self.action_space!r}"
            )
        if self.action_space.startswith("joint_delta") and not self.use_state:
            raise ValueError(
                "joint_delta 는 출력에 state 를 더해 행동을 만든다. use_state=false 로는 "
                "학습 목표와 추론이 어긋난다"
            )
        feat = int(m["feature_dim"])

        if self.encoder_mode == "shared":
            enc = ConvEncoder(m["encoder_channels"], feat)
            self.encoders = nn.ModuleDict({c: enc for c in self.camera_names})
        elif self.encoder_mode == "separate":
            self.encoders = nn.ModuleDict(
                {c: ConvEncoder(m["encoder_channels"], feat) for c in self.camera_names}
            )
        else:
            raise ValueError(f"encoder_mode 는 separate|shared 여야 한다: {self.encoder_mode}")

        in_dim = feat * len(self.camera_names) + (state_dim if self.use_state else 0)
        dims = [in_dim, *[int(d) for d in m["hidden_dims"]]]
        head: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            head += [nn.Linear(a, b), nn.ReLU(inplace=True), nn.Dropout(float(m["dropout"]))]
        head.append(nn.Linear(dims[-1], action_dim))
        self.head = nn.Sequential(*head)

    def forward(self, images: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        """Raw head output. Under `joint_delta` this is the residual, not the action.
        헤드의 원 출력. `joint_delta` 에서는 이것이 행동이 아니라 잔차다.

        Converting here would hide which space the loss is computed in. The caller
        applies `to_action` when it wants an action.
        여기서 변환하면 손실이 어느 공간에서 계산되는지가 가려진다. 행동이 필요한
        호출자가 `to_action` 을 적용한다."""
        feats = [self.encoders[c](images[c]) for c in self.camera_names]
        if self.use_state:
            feats.append(state)
        return self.head(torch.cat(feats, dim=1))

    def n_params(self) -> int:
        """Trainable parameter count — Jetson budget depends on this.
        학습 파라미터 수. Jetson 예산이 여기 걸린다."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@dataclass
class CheckpointMeta:
    """What a checkpoint must carry to be interpretable later.
    체크포인트가 나중에 해석되려면 함께 지녀야 하는 것."""

    camera_names: list[str]
    contract_version: str
    # Which space the head regresses. A checkpoint without this predates the
    # delta target and is absolute -- loading it as delta would add the state
    # twice and drive the arm to nonsense.
    # 헤드가 어느 공간을 회귀하는가. 이 값이 없는 체크포인트는 델타 목표 이전 것이라
    # 절대다. 델타로 불러오면 상태가 두 번 더해져 팔이 엉뚱하게 간다.
    action_space: str
    # Dataset statistics the head was standardised against. Without them the raw
    # output cannot be turned back into an action.
    # 헤드가 표준화된 기준이 된 데이터셋 통계. 이게 없으면 원 출력을 행동으로
    # 되돌릴 수 없다.
    target_mean: list[float] | None
    target_std: list[float] | None
    train_config: dict[str, Any]
    config_sha: str
    code_sha: str
    n_params: int
    n_episodes: int
    n_samples: int
    epochs_run: int
    best_val_loss: float
    trained_on: str  # "random_tensors" | dataset path
    # 이 체크포인트가 쓰는 그리퍼 규약. 없으면 시뮬 설정에서 파생된 값으로 읽는다
    # (2026-09-12 이전 체크포인트). 실물 데이터로 학습하면 여기 값이 다르다.
    gripper_open_norm: float | None = None
    gripper_close_norm: float | None = None
    note: str = (
        "val_loss 는 학습이 망가지지 않았는지 확인용이다. 성능 판정은 "
        "tools/eval_rollout.py 의 롤아웃 성공률이 한다."
    )


def save_checkpoint(path: Path, model: BCNet, meta: CheckpointMeta) -> Path:
    """Write weights and the metadata needed to load them correctly.
    가중치와, 그것을 올바로 불러오는 데 필요한 메타데이터를 쓴다."""
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": asdict(meta)}, path)
    return path


class BCPolicy:
    """A trained BC checkpoint behind the `Policy` protocol.
    학습된 BC 체크포인트를 `Policy` 프로토콜 뒤에 둔 것.

    It sees only an `Observation` and returns an action in contract units, so
    the rollout harness scores it under exactly the same conditions as the
    baselines. Nothing about it is privileged.
    `Observation` 만 보고 계약 단위의 행동을 돌려준다. 그래서 롤아웃 harness 가
    baseline 과 **정확히 같은 조건**으로 채점한다. 특권 정보는 없다.
    """

    uses_privileged_state = False

    def __init__(self, ckpt_path: Path, device: str = "cpu") -> None:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.meta = blob["meta"]
        self.device = torch.device(device)
        # 체크포인트가 자기 그리퍼 규약을 지니면 그것을 쓴다. 없으면 설정 파생값이다.
        _o = self.meta.get("gripper_open_norm")
        _c = self.meta.get("gripper_close_norm")
        self._grip_norms: tuple[float, float] | None = (
            (float(_o), float(_c)) if _o is not None and _c is not None else None
        )
        self.model = BCNet(self.meta["camera_names"], self.meta["train_config"])
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(self.device).eval()
        d = self.meta["train_config"]["data"]
        self._mean = float(d["image_mean"])
        self._std = float(d["image_std"])
        self.action_space = str(self.meta.get("action_space", "joint_absolute"))
        tm, ts = self.meta.get("target_mean"), self.meta.get("target_std")
        self._t_mean = (
            torch.tensor(tm, dtype=torch.float32, device=self.device) if tm else None
        )
        self._t_std = (
            torch.tensor(ts, dtype=torch.float32, device=self.device) if ts else None
        )
        self._ckpt = Path(ckpt_path)
        # The head is linear, so the network can predict outside the contract's
        # [-1, 1]. Clipping keeps the action valid, and counting how often it
        # happens is a real signal: a model that constantly saturates has not
        # learned the action distribution.
        # 헤드가 선형이라 계약 범위 [-1, 1] 밖을 예측할 수 있다. 클립하면 행동은
        # 유효해지고, 얼마나 자주 그러는지 세는 것은 실제 신호다 — 계속 포화되는
        # 모델은 행동 분포를 배우지 못한 것이다.
        self.n_actions = 0
        self.n_clipped = 0

    @property
    def name(self) -> str:
        return "bc"

    def reset(self, seed: int | None = None) -> None:
        """BC is memoryless — only the clipping counters reset.
        BC 는 상태가 없다. 클립 카운터만 초기화한다."""
        self.n_actions = 0
        self.n_clipped = 0
        return None

    @torch.no_grad()
    def act(self, obs: Observation) -> np.ndarray:
        """One observation in, one contract-unit action out.
        관측 하나 받아 계약 단위 행동 하나를 낸다."""
        images = {}
        for cam in self.model.camera_names:
            if cam not in obs.images:
                raise KeyError(
                    f"체크포인트는 카메라 {self.model.camera_names} 를 기대하는데 "
                    f"관측에는 {sorted(obs.images)} 만 있다"
                )
            arr = torch.from_numpy(obs.images[cam].astype(np.float32) / 255.0)
            images[cam] = ((arr - self._mean) / self._std).unsqueeze(0).to(self.device)
        state = torch.from_numpy(np.asarray(obs.state, dtype=np.float32)).unsqueeze(0)
        state = state.to(self.device)
        raw = self.model(images, state)
        # 헤드의 원 출력을 남긴다. `joint_delta_gripper_binary` 에서 그리퍼 채널은
        # 로짓이고, `to_action` 이 부호만 남기고 크기를 버린다 — 계측에는 크기가
        # 필요하다 (조건부 확률이 0.5 아래에 눌렸는지 요동치는지가 처방을 가른다).
        # 동작에는 영향이 없다. 읽기만 한다.
        self.last_raw = raw.squeeze(0).detach().cpu().numpy().astype(np.float64)
        out = to_action(raw, state, self.action_space, self._t_mean, self._t_std,
                        self._grip_norms)
        action = out.squeeze(0).cpu().numpy().astype(np.float32)
        action = check_action(action, self.name)
        clipped = np.clip(action, -1.0, 1.0)
        self.n_actions += 1
        self.n_clipped += int(np.any(clipped != action))
        return clipped

    def describe(self) -> str:
        """One line naming what this checkpoint actually is.
        이 체크포인트가 실제로 무엇인지 한 줄로."""
        m = self.meta
        return (
            f"bc ckpt {self._ckpt.name} · 행동공간 {self.action_space}"
            f"{'(표준화)' if self._t_std is not None else ''} · "
            f"파라미터 {m['n_params']:,} · "
            f"학습대상 {m['trained_on']} · 에피소드 {m['n_episodes']} · "
            f"샘플 {m['n_samples']} · epochs {m['epochs_run']} · "
            f"best val_loss {m['best_val_loss']:.5f}"
        )

    def clip_report(self) -> str:
        """How often the raw prediction left the contract range.
        원 예측이 계약 범위를 벗어난 빈도."""
        if self.n_actions == 0:
            return "행동 없음"
        pct = 100.0 * self.n_clipped / self.n_actions
        verdict = " — **포화가 잦다. 행동 분포를 배우지 못했을 수 있다**" if pct > 20 else ""
        return f"계약 범위 초과로 클립된 행동 {self.n_clipped}/{self.n_actions} ({pct:.1f}%){verdict}"
