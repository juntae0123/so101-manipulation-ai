#!/usr/bin/env python3
"""Corrupt what a policy sees at inference time, leaving training untouched.
추론 시점에 정책이 보는 것을 망가뜨린다. 학습은 건드리지 않는다.

왜 따로 있나 — 2026-09-15 에 내 어블레이션의 정체를 잘못 알고 있었다 🔴.

    `--image-noise` 는 `eval/repeat.py` 에서 **학습 명령에만** 붙는다.
    즉 지금까지의 "이미지 어블레이션" 은 **노이즈로 학습하고 정상 이미지로 평가**한
    것이었다. 그것도 타당한 검사지만, 답하는 질문이 다르다.

        노이즈 학습 -> 정상 평가   "학습이 이미지에서 무언가를 배우는가"
        정상 학습   -> 노이즈 평가  "학습된 정책이 추론할 때 이미지를 참조하는가"

    앞의 것은 val 도 같이 무너지므로 **"이미지 미사용" 과 "학습 자체가 망가짐" 이
    섞인다.** 뒤의 것은 학습이 정상인 체크포인트를 그대로 쓰므로 그 교란이 없다.

이 모듈은 뒤쪽을 가능하게 한다. 정책을 감싸기만 하므로 롤아웃 루프는 그대로다.

⚠️ `blackout` 은 `noise` 보다 **약한** 검사일 수 있다. 검은 화면은 매 틱 같은
   상수 입력이라 신경망이 "정보 없음" 으로 안정적으로 다룰 수 있다. 가우시안은
   틱마다 달라 더 파괴적이다. blackout 은 추가 조건이지 대체가 아니다.

⚠️ random crop/shift 는 여기 없다. 그것은 파괴가 아니라 **증강**이라 성능이 오를
   수도 있고, 그러면 어블레이션 해석이 꼬인다.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sim.base import Observation

# ⚠️ `Policy` 를 상속하지 않는다. 이 저장소의 정책은 전부 구조적 타이핑이고
#    (`ACTPolicy`·`HoldPolicy`·`ScriptedPickPolicy` 어느 것도 상속하지 않는다),
#    한 군데만 다르게 하면 Protocol 이 바뀔 때 여기만 조용히 깨진다.

MODES = ("noise", "blackout", "freeze")


class PerturbedObsPolicy:
    """Wrap a policy and corrupt its image input at inference time.
    정책을 감싸고 추론 시점의 이미지 입력을 망가뜨린다.

    학습된 가중치는 그대로다. 정책이 이미지를 실제로 참조한다면 성공률이 무너지고,
    상태만 보고 있었다면 거의 그대로다.
    """

    def __init__(self, inner, mode: str = "noise", sigma: float = 96.0) -> None:
        if mode not in MODES:
            raise ValueError(f"모드는 {MODES} 중 하나여야 한다: {mode}")
        self.inner = inner
        self.mode = mode
        self.sigma = float(sigma)
        self._rng = np.random.default_rng(0)
        self._first: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        """Same harness slot as the wrapped policy.
        감싼 정책과 같은 harness 자리다.

        ⚠️ 여기서 다른 이름을 돌려주면 `eval/rollout.py` 가 배포 게이트·실패모양·
        클립 통계를 찾지 못하고 **조용히 건너뛴다.** 0915 에 같은 실수로 8잡이 죽었다."""
        return self.inner.name

    @property
    def uses_privileged_state(self) -> bool:
        """Unchanged by perturbation.
        교란과 무관하다."""
        return getattr(self.inner, "uses_privileged_state", False)

    def reset(self, seed: int | None = None) -> None:
        """Reset the wrapped policy and re-seed the corruption.
        감싼 정책을 리셋하고 교란 난수를 다시 시드한다.

        시드를 고정하지 않으면 같은 조건을 두 번 돌렸을 때 값이 달라져 재현이 깨진다."""
        self.inner.reset(seed)
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self._first = {}

    def _corrupt(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return corrupted copies. Never mutates the caller's arrays.
        망가뜨린 사본을 돌려준다. 호출자의 배열을 건드리지 않는다."""
        out: dict[str, np.ndarray] = {}
        for cam, img in images.items():
            if self.mode == "blackout":
                out[cam] = np.zeros_like(img)
            elif self.mode == "freeze":
                # 첫 프레임을 계속 준다 — "시간에 따라 변하는 시각 정보" 만 제거한다.
                # 정적 배치 정보는 남으므로 noise 보다 약하고, 그 차이가 정보다.
                out[cam] = self._first.setdefault(cam, img.copy())
            else:
                noisy = img.astype(np.float32) + self._rng.normal(
                    0.0, self.sigma, size=img.shape
                )
                out[cam] = np.clip(noisy, 0, 255).astype(img.dtype)
        return out

    def act(self, obs: Observation) -> np.ndarray:
        """Corrupt the images, then defer to the wrapped policy.
        이미지를 망가뜨린 뒤 감싼 정책에 넘긴다."""
        return self.inner.act(replace(obs, images=self._corrupt(obs.images)))
