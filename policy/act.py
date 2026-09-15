"""ACT-lite: the BC backbone with a k-step action chunk head.
ACT-lite: BC 백본에 k 스텝 행동 청크 헤드만 붙인 것.

## 왜 백본을 그대로 두나

`policy/bc.py` 의 `BCNet` docstring 이 이미 적어놓은 이유다 — 백본과 청킹을 동시에
바꾸면 수치 차이를 무엇 때문인지 귀속시킬 수 없다. 여기서 바뀌는 것은 **출력
차원 하나**다.

    BCNet   head -> (B, 6)
    ACTNet  head -> (B, K*6) -> (B, K, 6)

`chunk=1` 이면 `action_dim = 6` 이라 **BCNet 과 구조·파라미터 수·state_dict 키가
완전히 같다.** 그래서 같은 시드로 학습하면 BC 와 비트 단위로 같은 결과가 나와야
하고, 그것이 계측기 검증(G0)이다. 안 맞으면 청킹 구현이 다른 것을 하고 있다.

## 앵커 규약

청크의 모든 스텝은 **같은 현재 상태**를 기준으로 한다.

    target[i] = f(action[t+i], state[t])        i = 0 .. K-1

`state[t+i]` 를 기준으로 하면 추론 시 미래 상태를 알아야 해서 계산 자체가 불가능하다.
트랙 A 의 `SPEC_umi_relative_target_trajectory_0914.md` 도 같은 규약이다
("모든 미래 target은 직전 target이 아니라 같은 현재 pinch pose를 anchor로 한다").

## 실행 규약

한 관측에서 K 스텝을 예측하고 **K 스텝에 걸쳐 그대로 실행**한다 (청크 안은 열린 루프).
그래서 `lead-k` 단일-step 과 달리 **제어 게인이 바뀌지 않는다** — 0914 의 lead 스윕이
가진 교란(k틱 변위를 1틱 목표로 전송)이 여기에는 없다.

⚠️ 청킹은 되먹임 고리를 끊을 뿐, **타깃이 외삽으로 풀린다는 성질은 없애지 못한다.**
오히려 "관측 한 번 보고 가던 대로 K칸"이라는 지름길을 만들 수 있다. 그래서
사전등록(`docs/PREREG_chunking_0915.md`)의 G4 가 이미지 어블레이션을 요구한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from contract.episode import RANGE_TOLERANCE
from policy.base import check_action
from sim.base import Observation
from policy.bc import (
    _GRIPPER,
    BCNet,
    CheckpointMeta,
    to_action,
    training_target,
)


class ACTNet(BCNet):
    """BCNet with a K-step head. `chunk=1` reduces to BCNet exactly.
    K 스텝 헤드를 단 BCNet. `chunk=1` 이면 BCNet 과 정확히 같다."""

    def __init__(
        self,
        camera_names: list[str],
        cfg: dict[str, Any],
        state_dim: int = 6,
        action_dim: int = 6,
        chunk: int = 1,
    ) -> None:
        if chunk < 1:
            raise ValueError(f"chunk 는 1 이상이어야 한다: {chunk}")
        # 헤드 출력만 K 배로 늘린다. chunk=1 이면 BCNet 과 동일한 인자다.
        super().__init__(camera_names, cfg, state_dim=state_dim,
                         action_dim=action_dim * chunk)
        self.chunk = int(chunk)
        self.step_dim = int(action_dim)

    def forward(
        self, images: dict[str, torch.Tensor], state: torch.Tensor
    ) -> torch.Tensor:
        """Raw head output. (B, D) when chunk=1, else (B, K, D).
        헤드의 원 출력. chunk=1 이면 (B, D), 아니면 (B, K, D).

        ⚠️ chunk=1 에서 `(B, 1, D)` 를 내지 않는 이유: 그러면 손실·타깃·체크포인트
        경로가 전부 BC 와 한 축 어긋나고, **계측기 검증(G0)이 "같다"를 확인할 수
        없게 된다.** chunk=1 은 기존 BC 와 구분 불가능해야 한다."""
        flat = super().forward(images, state)
        if self.chunk == 1:
            return flat
        return flat.view(flat.shape[0], self.chunk, self.step_dim)


def chunk_training_target(
    action_chunk: torch.Tensor,
    state: torch.Tensor,
    action_space: str,
    grip_mid: float | None = None,
) -> torch.Tensor:
    """Loss target for a chunk, every step anchored on the same current state.
    청크의 손실 목표. 모든 스텝이 **같은 현재 상태**를 기준으로 한다.

    action_chunk (B, K, D) · state (B, D) -> (B, K, D)

    `training_target` 은 `[..., _GRIPPER]` 로 인덱싱하므로 청크 축이 있어도 그대로
    쓸 수 있다. 여기서는 state 를 브로드캐스트해 앵커 규약만 강제한다."""
    if action_chunk.ndim != 3:
        raise ValueError(f"action_chunk 는 (B, K, D) 여야 한다: {tuple(action_chunk.shape)}")
    if state.ndim != 2:
        raise ValueError(f"state 는 (B, D) 여야 한다: {tuple(state.shape)}")
    return training_target(action_chunk, state.unsqueeze(1), action_space, grip_mid)


def chunk_target_scale(
    target: torch.Tensor,
    action_space: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(step, joint) mean and std of the chunk target.
    청크 목표의 (스텝, 관절)별 평균과 표준편차.

    스텝별로 따로 재는 이유: 멀리 있는 스텝일수록 델타가 크다. 하나의 스케일로
    묶으면 가까운 스텝이 눌린다.

    ⚠️ `bc.target_scale` 은 `mean[_GRIPPER]` 로 인덱싱하는데, 청크 목표에서 그
    축은 **스텝 축**이다. 그대로 쓰면 엉뚱한 자리를 덮어쓴다. 그래서 여기서
    `[..., _GRIPPER]` 로 다시 쓴다."""
    if target.ndim != 3:
        raise ValueError(f"target 은 (N, K, D) 여야 한다: {tuple(target.shape)}")
    mean = target.mean(dim=0)
    std = target.std(dim=0)
    # ⚠️ 2026-09-15 정정 — 하한이 1e-8 이었다. 계약의 RANGE_TOLERANCE 는 1e-4 이고,
    # 그보다 작은 움직임은 **계약상 의미 없는 값**이다. 그런데 std 1e-5 인 축을
    # 그대로 나누면 잔차가 1e5 배로 증폭된다. DAgger 데이터의 `wrist_roll` 이
    # std 1e-05·90% 가 정확히 0 이었고, 그 한 축 때문에 표준화 손실이 12~14 로
    # 터졌다 (자명한 예측기 1.66 의 7배). train 부터 발산했다 🟢.
    # 움직이지 않는 축은 표준화하지 않는다 — 상수 목표에 스케일은 의미가 없다.
    std = torch.where(std < RANGE_TOLERANCE, torch.ones_like(std), std)
    if action_space == "joint_delta_gripper_binary":
        # 그리퍼 채널은 0/1 라벨이고 헤드 출력은 로짓이다. 표준화하면 BCE 가 보는
        # 라벨이 0/1 이 아니게 된다. 항등으로 둔다 (bc.target_scale 과 같은 이유).
        mean = mean.clone()
        std = std.clone()
        mean[..., _GRIPPER] = 0.0
        std[..., _GRIPPER] = 1.0
    return mean, std


def chunk_to_action(
    raw: torch.Tensor,
    state: torch.Tensor,
    action_space: str,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    grip_norms: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Network output -> K contract-unit actions, all anchored on `state`.
    신경망 출력 -> 계약 단위 행동 K 개. 전부 `state` 를 기준으로 한다.

    raw (B, K, D) · state (B, D) -> (B, K, D)"""
    if raw.ndim != 3:
        raise ValueError(f"raw 는 (B, K, D) 여야 한다: {tuple(raw.shape)}")
    return to_action(raw, state.unsqueeze(1), action_space,
                     target_mean, target_std, grip_norms)


@dataclass
class ACTCheckpointMeta(CheckpointMeta):
    """CheckpointMeta plus the chunk length. Without it the head cannot be reshaped.
    CheckpointMeta 에 청크 길이를 더한 것. 없으면 헤드를 다시 펼 수 없다."""

    chunk: int = 1
    # 청크 안에서 실제로 실행하는 스텝 수. K 보다 작으면 나머지는 버리고 다시 관측한다.
    # 기본은 K (완전 열린 루프). 줄이면 되먹임이 잦아지는 대신 청킹 효과가 준다.
    execute: int = 0


def save_act_checkpoint(path: Path, model: ACTNet, meta: ACTCheckpointMeta) -> Path:
    """Write weights plus the metadata needed to reload them.
    가중치와 다시 불러오는 데 필요한 메타데이터를 쓴다."""
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": asdict(meta)}, path)
    return path


class ACTPolicy:
    """A trained chunk checkpoint behind the `Policy` protocol.
    학습된 청크 체크포인트를 `Policy` 프로토콜 뒤에 둔 것.

    `act()` 는 버퍼가 비었을 때만 관측을 보고 K 스텝을 한 번에 만든다. 그 뒤
    `execute` 스텝 동안은 **관측을 보지 않고** 버퍼에서 꺼낸다. 이것이 청킹이
    끊는 되먹임 고리다.

    ⚠️ 버퍼에 담기는 것은 계약 단위의 **절대 행동**이다. 앵커 상태를 예측 시점에
    이미 더했으므로, 실행 중 상태가 변해도 목표가 따라 움직이지 않는다. 그것이
    열린 루프의 정의다.
    """

    uses_privileged_state = False

    def __init__(self, ckpt_path: Path, device: str = "cpu") -> None:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.meta = blob["meta"]
        self.device = torch.device(device)
        self.chunk = int(self.meta.get("chunk", 1))
        ex = int(self.meta.get("execute", 0) or 0)
        self.execute = self.chunk if ex <= 0 else min(ex, self.chunk)

        _o = self.meta.get("gripper_open_norm")
        _c = self.meta.get("gripper_close_norm")
        self._grip_norms: tuple[float, float] | None = (
            (float(_o), float(_c)) if _o is not None and _c is not None else None
        )
        self.model = ACTNet(
            self.meta["camera_names"], self.meta["train_config"], chunk=self.chunk
        )
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
        self._buffer: list[np.ndarray] = []
        # 클립 통계는 BCPolicy 와 같은 의미다. 계속 포화되면 행동 분포를 못 배운 것이다.
        self.n_actions = 0
        self.n_clipped = 0
        # 청킹 특유의 계측: 관측을 몇 번이나 실제로 봤는가.
        self.n_predictions = 0

    @property
    def name(self) -> str:
        """Always "bc" — this is the harness slot, not the model family.
        항상 "bc" 다. 이것은 모델 종류가 아니라 **평가 harness 안의 "학습 정책" 자리**다.

        ⚠️ 2026-09-15 정정 — 여기서 chunk>1 일 때 "act" 를 돌려줬다가 8잡이 전부
        죽었다. `eval/repeat.py` 가 `success_rates["bc"]` 로 읽고,
        `eval/rollout.py` 는 **배포 게이트(floor/chance)·실패모양 리포트·클립 통계**를
        전부 이 키로 찾는다. 이름이 달라지면 repeat 은 KeyError 로 죽고 rollout 은
        **게이트를 조용히 건너뛴다** — 후자가 훨씬 나쁘다. 수치는 그대로 나오는데
        판정만 사라진다.

        청크 길이는 이름이 아니라 `describe()`·체크포인트 메타·EXP_LOG 조건의
        `chunk` 키에 남는다. 기록은 하나도 잃지 않는다."""
        return "bc"

    def reset(self, seed: int | None = None) -> None:
        """Clear the chunk buffer. A stale buffer would leak the previous episode.
        청크 버퍼를 비운다. 남아 있으면 직전 에피소드가 새어 들어간다."""
        self._buffer = []
        self.n_actions = 0
        self.n_clipped = 0
        self.n_predictions = 0
        return None

    @torch.no_grad()
    def _predict(self, obs: Observation) -> None:
        """Look at one observation, fill the buffer with `execute` actions.
        관측 하나를 보고 버퍼를 `execute` 개로 채운다."""
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
        raw = self.model(images, state)                      # (1, D) 또는 (1, K, D)
        if raw.ndim == 2:                                    # chunk=1 — BC 경로 그대로
            out = to_action(raw, state, self.action_space,
                            self._t_mean, self._t_std, self._grip_norms)
            seq = out.cpu().numpy().astype(np.float32)        # (1, D)
        else:
            out = chunk_to_action(raw, state, self.action_space,
                                  self._t_mean, self._t_std, self._grip_norms)
            seq = out.squeeze(0).cpu().numpy().astype(np.float32)  # (K, D)
        self._buffer = [seq[i].copy() for i in range(self.execute)]
        self.n_predictions += 1

    def act(self, obs: Observation) -> np.ndarray:
        """One action per tick, taken from the buffer.
        틱마다 하나씩, 버퍼에서 꺼내 낸다."""
        if not self._buffer:
            self._predict(obs)
        action = self._buffer.pop(0)
        action = check_action(action, self.name)
        clipped = np.clip(action, -1.0, 1.0)
        self.n_actions += 1
        self.n_clipped += int(np.any(clipped != action))
        return clipped

    def describe(self) -> str:
        """One line for the run record.
        실행 기록용 한 줄."""
        return (
            f"{self.name} chunk={self.chunk} execute={self.execute} "
            f"space={self.action_space} ckpt={self._ckpt.name}"
        )

    def clip_report(self) -> str:
        """How often the head saturated, and how often it looked.
        헤드가 얼마나 포화됐는지, 그리고 관측을 몇 번 봤는지."""
        if self.n_actions == 0:
            return "행동 0건"
        return (
            f"클립 {self.n_clipped}/{self.n_actions} "
            f"({self.n_clipped / self.n_actions:.1%}) · "
            f"관측 {self.n_predictions}회 (틱당 {self.n_predictions / self.n_actions:.2f})"
        )


def load_policy(ckpt_path: Path, device: str = "cpu"):
    """Pick the right policy class by what the checkpoint says its chunk is.
    체크포인트가 말하는 청크 길이에 따라 정책 클래스를 고른다.

    청크 길이는 **체크포인트에만** 적혀 있다. 호출자가 정하게 두면 학습과 평가가
    다른 K 로 갈리고, 그 어긋남은 예외 없이 조용히 틀린 롤아웃으로 나타난다.

    `chunk` 가 없거나 1 인 체크포인트는 `BCPolicy` 로 간다 — 기존 경로를 그대로
    쓰기 위해서다. `ACTPolicy(chunk=1)` 이 `BCPolicy` 와 행동 단위로 동일함은
    확인했지만(2026-09-15 🟢), 기존 수치가 나온 코드 경로를 바꾸지 않는 쪽이
    비교를 깨뜨리지 않는다."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    chunk = int(blob.get("meta", {}).get("chunk", 1) or 1)
    if chunk > 1:
        return ACTPolicy(ckpt_path, device=device)
    from policy.bc import BCPolicy

    return BCPolicy(ckpt_path, device=device)
