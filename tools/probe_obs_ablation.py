"""Does the policy actually use the images? Closed loop.
정책이 이미지를 실제로 쓰는가. 폐루프.

    python tools/probe_obs_ablation.py --help

## 왜

"팔이 부정확하다" 와 **"팔이 물체를 아예 안 본다"** 는 밖에서 똑같이 생겼다 —
둘 다 낮은 성공률이다. 그런데 처방이 정반대다. 이미지가 기여하지 않으면 정책은
관절각만의 함수이고 평균 궤적을 재생할 뿐이라, 행동 표현·손실 가중·데이터 증량이
전부 무의미해진다.

`eval/image_sensitivity.py` 는 **개루프**(교사강요) 민감도를 잰다. 이 도구는 **폐루프**다 —
이미지를 바꾸고 롤아웃을 끝까지 돌려 **성공률**이 얼마나 떨어지는지 본다. 개루프
민감도가 0 이 아니어도 루프를 닫기엔 부족할 수 있다.

## 주 판정은 `blank` 가 아니라 `donor` 다

`blank`(0 채움)는 정규화 후 평균에서 멀어 **분포 밖**이다. 거기서 성공률이 떨어져도
"이미지를 쓴다"가 아니라 "이상한 입력에 망가진다"일 수 있다.
`donor` 는 다른 에피소드의 **같은 틱** 이미지다 — 분포 안이면서 **물체 위치만 틀리다.**
그래서 주 판정은 `donor_both` 이고 `blank` 는 보조로만 읽는다.

판정 기준은 `docs/PREREG_obs_ablation_0911.md` 에 결과 보기 전에 박았다.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402

runtime_limits.claim("probe_obs_ablation")
runtime_limits.torch_threads()

import numpy as np  # noqa: E402

from policy.bc import BCPolicy  # noqa: E402
from sim.base import Observation  # noqa: E402
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from sim.mujoco.env import MujocoPickEnv  # noqa: E402
from tracking.exp_log import code_digest, file_digest, log_run  # noqa: E402

CODE_SHA = code_digest()
CONDITIONS = (
    "full",
    "donor_both",
    "donor_front",
    "donor_wrist",
    "freeze_both",
    "blank_both",
)


class Ablator:
    """Replace camera content before the policy sees it. State is untouched.
    정책이 보기 전에 카메라 내용을 바꾼다. 상태는 건드리지 않는다.

    한 번에 하나만 바꾼다 — 상태·행동·환경은 그대로다. 그래야 성공률 차이를
    이미지에 귀속시킬 수 있다.
    """

    def __init__(self, base: BCPolicy, condition: str, front: str, wrist: str,
                 donor: list[dict[str, np.ndarray]] | None) -> None:
        self.base = base
        self.condition = condition
        self._front = front
        self._wrist = wrist
        self._donor = donor
        self._first: dict[str, np.ndarray] | None = None
        self._t = 0

    def reset(self, seed: int) -> None:
        self.base.reset(seed=seed)
        self._first = None
        self._t = 0

    def _targets(self) -> tuple[str, ...]:
        if self.condition.endswith("_both"):
            return (self._front, self._wrist)
        if self.condition.endswith("_front"):
            return (self._front,)
        if self.condition.endswith("_wrist"):
            return (self._wrist,)
        return ()

    def act(self, obs: Observation) -> np.ndarray:
        if self._first is None:
            self._first = {k: v.copy() for k, v in obs.images.items()}

        if self.condition != "full":
            images = dict(obs.images)
            for cam in self._targets():
                if self.condition.startswith("blank"):
                    images[cam] = np.zeros_like(obs.images[cam])
                elif self.condition.startswith("freeze"):
                    images[cam] = self._first[cam]
                elif self.condition.startswith("donor"):
                    if self._donor is None:
                        raise RuntimeError("donor 캐시가 없다")
                    # 기증자가 먼저 끝났으면 마지막 프레임을 쓴다.
                    frame = self._donor[min(self._t, len(self._donor) - 1)]
                    images[cam] = frame[cam]
                else:
                    raise ValueError(self.condition)
            obs = replace(obs, images=images)

        self._t += 1
        return self.base.act(obs)


def rollout(env: MujocoPickEnv, policy: Any, seed: int) -> dict[str, Any]:
    """One episode under the standard conditions. Stops on first success.
    표준 조건의 에피소드 하나. 최초 성공 시 종료."""
    obs = env.reset(seed=seed)
    policy.reset(seed)
    success = False
    ticks = 0
    nearest = float("inf")
    for _ in range(env.max_ticks):
        obs = env.step(policy.act(obs))
        ticks += 1
        xy, _ = env.pinch_to_object_m()
        nearest = min(nearest, float(xy))
        if env.is_success():
            success = True
            break
    return {"seed": seed, "success": success, "ticks": ticks,
            "nearest_xy_mm": nearest * 1000.0}


def cache_donor(env: MujocoPickEnv, policy: BCPolicy, seed: int) -> list[dict[str, np.ndarray]]:
    """Record one episode's per-tick images to use as in-distribution wrong input.
    분포 안의 '틀린 입력' 으로 쓸 에피소드 하나의 틱별 이미지를 기록한다.

    평가 시드 블록 **밖**의 시드를 쓴다 — 평가에 누수되면 안 된다.
    """
    obs = env.reset(seed=seed)
    policy.reset(seed)
    frames: list[dict[str, np.ndarray]] = []
    for _ in range(env.max_ticks):
        frames.append({k: v.copy() for k, v in obs.images.items()})
        obs = env.step(policy.act(obs))
        if env.is_success():
            break
    return frames


def verdict(drop_pp: float, base_rate: float, near_ratio: float) -> str:
    """Pre-registered read of the donor_both drop, guarded against a floor effect.
    `donor_both` 하락폭에 대한 사전등록 판정. 바닥 효과를 막는다.

    **정정 (2026-09-11, 실행 후).** 초판은 절대 하락폭만 봤다. 원래 성공률이 낮은
    체크포인트는 **떨어질 자리가 없어서** 하락폭이 작게 나오고, 실제로는 100% 를
    잃었는데도 "이미지를 안 쓴다"로 찍혔다 — seed0(6/100)에서 그렇게 오판했다.
    사전등록 예측 4번에 "바닥 효과 때문에 절대 %p 로 읽는다"고 적어놓고 함수에
    반영하지 못한 것이 원인이다.

    이제 세 가지를 함께 본다:
      - 절대 하락폭 (원래 기준)
      - **상대 상실률** — 바닥 효과를 드러낸다
      - **최근접 거리 배율** — 성공률이 0 이어도 팔이 얼마나 멀어졌는지는 남는다
    """
    relative = (drop_pp / (100.0 * base_rate)) if base_rate > 0 else float("nan")
    strong = drop_pp >= 50.0 or (relative >= 0.8 and near_ratio >= 3.0)
    if strong:
        return "images_used__arm_precision_is_the_real_bottleneck"
    if drop_pp >= 20.0 or (relative >= 0.5 and near_ratio >= 2.0):
        return "partial_use__improve_observation_and_precision_together"
    if base_rate < 0.15:
        return "inconclusive__floor_effect__read_relative_loss_and_nearest_ratio"
    return "images_barely_used__bottleneck_is_observation_not_precision"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-ckpt", type=Path, action="append", required=True)
    ap.add_argument("--expected", type=int, action="append", default=[],
                    help="`full` 조건의 기존 성공 수. 계측기 재현 검사")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=3000)
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--policy-device", type=str, default="cpu")
    ap.add_argument("--front-camera", type=str, default="cam_front")
    ap.add_argument("--wrist-camera", type=str, default="cam_wrist")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--author", type=str, default="김준태(트랙B)")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        raise FileExistsError(args.out)
    if args.expected and len(args.expected) != len(args.policy_ckpt):
        ap.error("--expected 개수가 --policy-ckpt 개수와 다르다")

    cfg = load_config()
    seeds = [args.seed_base + i for i in range(args.episodes)]
    donor_seed = args.seed_base - 1  # 평가 블록 밖

    print("관측 절제 계측 (폐루프) — 정책이 이미지를 쓰는가")
    print(f"조건: {args.episodes}편 · seeds {seeds[0]}~{seeds[-1]} · render · "
          f"policy-device {args.policy_device} · jitter ±{args.jitter * 1000:.0f}mm")
    print(f"기증자 시드 {donor_seed} (평가 블록 밖) · sim cfg sha {file_digest(DEFAULT_CONFIG)}")
    print("⚠️ 주 판정은 donor 다. blank 는 분포 밖이라 보조로만 읽는다\n")

    results: dict[str, Any] = {}
    with MujocoPickEnv(cfg=cfg, render=True, object_jitter_m=args.jitter,
                       max_ticks=200) as env:
        cams = env.camera_names
        for cam in (args.front_camera, args.wrist_camera):
            if cam not in cams:
                raise ValueError(f"카메라 {cam} 이 환경에 없다. 있는 것: {cams}")

        for idx, ckpt in enumerate(args.policy_ckpt):
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
            base = BCPolicy(ckpt, device=args.policy_device)
            label = ckpt.stem
            print(f"===== {label} =====")
            donor = cache_donor(env, base, donor_seed)
            print(f"  기증자 캐시 {len(donor)}틱")

            per_condition: dict[str, Any] = {}
            for cond in CONDITIONS:
                pol = Ablator(base, cond, args.front_camera, args.wrist_camera, donor)
                eps = [rollout(env, pol, s) for s in seeds]
                hits = sum(e["success"] for e in eps)
                near = float(np.median([e["nearest_xy_mm"] for e in eps]))
                per_condition[cond] = {
                    "successes": hits,
                    "success_rate": hits / args.episodes,
                    "nearest_xy_mm_median": near,
                }
                print(f"  {cond:<13} {hits:>3}/{args.episodes}  "
                      f"최근접 중앙 {near:6.1f}mm", flush=True)

                if cond == "full" and args.expected:
                    want = args.expected[idx]
                    if hits != want:
                        raise RuntimeError(
                            f"instrument invalid: {label} full "
                            f"{hits}/{args.episodes}, 기대 {want}. "
                            "git_rev·config sha·체크포인트 경로를 먼저 대조하라"
                        )
                    print(f"  full 재현 fixture PASS ({hits})")

            base_rate = per_condition["full"]["success_rate"]
            drops = {
                cond: 100.0 * (base_rate - per_condition[cond]["success_rate"])
                for cond in CONDITIONS if cond != "full"
            }
            near_full = per_condition["full"]["nearest_xy_mm_median"]
            near_donor = per_condition["donor_both"]["nearest_xy_mm_median"]
            near_ratio = (near_donor / near_full) if near_full > 0 else float("nan")
            relative = (
                drops["donor_both"] / (100.0 * base_rate) if base_rate > 0 else float("nan")
            )
            v = verdict(drops["donor_both"], base_rate, near_ratio)
            results[label] = {
                "conditions": per_condition,
                "drop_pp": drops,
                "donor_both_relative_loss": relative,
                "donor_both_nearest_ratio": near_ratio,
                "verdict": v,
            }
            print("  하락폭(%p): " +
                  " · ".join(f"{k} {drops[k]:+.1f}" for k in drops))
            print(f"  donor_both 상대 상실 {100.0 * relative:.0f}% · "
                  f"최근접 {near_full:.1f} → {near_donor:.1f}mm ({near_ratio:.1f}배)")
            print(f"  판정 {v}")
            if drops["donor_front"] < 5.0 and base_rate >= 0.15:
                print(f"  ⚠️ {args.front_camera} 하락 {drops['donor_front']:.1f}%p "
                      "— 기여가 없다. 계약의 카메라 2대 요구(L62)에 영향")
            elif drops["donor_front"] < 5.0:
                print(f"  · {args.front_camera} 하락 {drops['donor_front']:.1f}%p 이지만 "
                      f"full 이 {100.0 * base_rate:.0f}% 라 바닥 효과다 — 기여 없음으로 읽지 않는다")

    payload = {
        "experiment": "obs_ablation",
        "conditions": {
            "episodes": args.episodes, "seed_base": args.seed_base,
            "seed_end": seeds[-1], "donor_seed": donor_seed,
            "jitter_m": args.jitter, "render": True,
            "policy_device": args.policy_device, "max_ticks": 200,
            "stop_on_first_success": True,
            "ablations": list(CONDITIONS),
            "front_camera": args.front_camera, "wrist_camera": args.wrist_camera,
            "policy_checkpoints": [str(c) for c in args.policy_ckpt],
            "sim_config_sha": file_digest(DEFAULT_CONFIG),
            "code_sha_at_launch": CODE_SHA,
        },
        "results": results,
        "interpretation_limit": (
            "Simulation only. Donor images never co-occurred with these states, "
            "so read the drop from full, not absolute values. blank is "
            "out-of-distribution and is a secondary reading only."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장 {args.out}")
    if args.log:
        rec = log_run(experiment="obs_ablation", author=args.author,
                      issue="S15P21A103-35", conditions=payload["conditions"],
                      result={"results": results,
                              "interpretation_limit": payload["interpretation_limit"]})
        print(f"EXP_LOG 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    print("판정은 사람이 한다. docs/PREREG_obs_ablation_0911.md 를 열고 대조하라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
