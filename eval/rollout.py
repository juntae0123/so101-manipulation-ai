"""Score policies by rollout success rate — the project's only real metric.
정책을 롤아웃 성공률로 채점한다. 이 프로젝트의 유일한 실제 지표다.

Validation loss is not this. In imitation learning the correlation between loss
and success is weak: loss measures similarity to the demonstrated trajectory,
but the robot succeeds by other paths too, and a low loss still accumulates
error into failure. Loss is only good for "did training break". This is the
number that decides anything.
validation loss 는 이게 아니다. 모방학습에서 손실과 성공률의 상관은 약하다.
손실은 시연 궤적과의 유사도를 재는데, 로봇은 다른 경로로도 성공하고 손실이
낮아도 오차가 누적돼 실패한다. 손실은 "학습이 망가졌나" 확인용이다.
판단을 내리는 수치는 이것이다.

Gate thresholds are set in GATES below, before any result was looked at.
게이트 기준은 아래 GATES 에 있고, 결과를 보기 전에 정했다.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


from sim.mujoco.env import MujocoPickEnv
from policy.base import Policy
from policy.baselines import (
    HoldPolicy,
    ReplayPolicy,
    ScriptedPickPolicy,
    ZeroPolicy,
)

from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config
from tracking.exp_log import file_digest, log_run


# Fixed before any number was produced. Changing these after seeing results is
# not a judgement, it is a rationalisation.
# 어떤 수치도 나오기 전에 확정했다. 결과를 보고 이걸 고치면 판정이 아니라
# 사후 합리화다.
# Reading rule for the failure shape, fixed before the first number. Not a gate
# -- it does not pass or fail a policy -- it says which of two different
# problems a failing policy has.
# 실패 형태를 읽는 규칙. 첫 수치가 나오기 전에 확정했다. 게이트가 아니다 — 정책을
# 통과/실패시키지 않는다. 실패하는 정책이 두 문제 중 어느 쪽인지 말한다.
PRECISION_NEAR_MM = 15.0   # 실측 재생 허용오차: ±10mm 3/4, ±15mm 1/4
PRECISION_FAR_MM = 30.0
# Grounded in the measured replay tolerance, not picked to make a result come out:
# ±5 mm 4/4, ±10 mm 3/4, ±15 mm 1/4 (MEASURE_baseline_0827).
# 실측 재생 허용오차에 맞춘 값이다. 결과가 나오게 하려고 고른 값이 아니다.
CLOSE_OK_MM = 5.0     # 닫는 순간 이 안이면 위치는 충분하다
CLOSE_BAD_MM = 10.0   # 이보다 멀면 닫는 순간의 위치 오차가 실패를 설명한다
# 지연(최근접 시점 → 닫는 시점)은 **판정 근거가 아니다.** 실측 2026-09-03 🟢:
# scripted 는 지연 +30틱(>5틱 100%)인데 84% 성공하고, bc 는 +17틱인데 7% 다.
# 최근접 시점은 접근 완료 시점이고 그 뒤 수십 틱은 하강·정렬 후 닫는 정상
# 시퀀스다. 차이는 그 사이에 거리를 유지하는가다 — scripted 4.7mm 유지,
# bc 5.1 -> 14.0mm 발산. 그래서 판정은 **닫는 순간의 절대 거리**로 한다.
# 지연은 참고값으로만 출력한다.
CLOSE_LAG_TICKS = 5   # 참고 표시용 문턱. 판정에 쓰지 않는다 (30Hz 기준 0.17s)
GRIP_SPAN_MIN = 0.05  # 명령 진폭이 이보다 작으면 닫는 동작 자체가 없다

# 닫힘으로 인정하려면 하위 문턱 아래로 내려가 이만큼 **연속으로 머물러야** 한다.
# 스크립트 정책은 계단식이라 아무 값이나 통하지만, 학습 정책은 연속 회귀 출력이라
# 중간값을 스치듯 지나갔다가 되돌아온다. 2026-09-07 실측 🟢: bc seed3000 은
# 틱 60 에 중간값(-0.478)을 -0.4957 로 스쳤다가 틱 66~84 에 -0.44→-0.25 로
# 되돌아왔고, 실제로 닫힌 것은 틱 102 였다 — 옛 정의는 42틱(1.4초) 일찍 찍었다.
CLOSE_HOLD_TICKS = 5      # 30Hz 기준 0.17초
CLOSE_LOW_FRAC = 0.25     # 진폭의 하위 25% 아래여야 "닫는 쪽"으로 본다
GATES: dict[str, str] = {
    "floor": "학습 정책 성공률 > hold 성공률 + 20%p. 못 넘으면 정책이 무의미하다.",
    "chance": "학습 정책 성공률 > zero 성공률 + 20%p. 못 넘으면 우연과 구분되지 않는다.",
    "task_validity": (
        "replay 성공률 < 30%. 이보다 높으면 고정 궤적으로 풀리는 태스크이므로 "
        "시각 정책이 학습할 것이 없다. 태스크를 다시 설계해야 한다."
    ),
    "ceiling": "scripted 성공률 >= 80%. 못 넘으면 태스크나 씬이 문제이지 정책 문제가 아니다.",
}


@dataclass
class RolloutResult:
    """Outcome of one episode under one policy.
    정책 하나로 돌린 에피소드 하나의 결과."""

    seed: int
    success: bool
    ticks: int
    lift_height_m: float
    object_xy: tuple[float, float]
    # Shape of the attempt, not just its outcome. Closest the pinch pocket came
    # to the object (mm, horizontal and 3-D), when, whether the jaws ever touched
    # it, and the lowest gripper command issued. These separate "went to the
    # wrong place" from "went to the right place and failed to close".
    # 결과가 아니라 시도의 **형태**. 파지 포켓이 물체에 가장 가까웠던 거리(mm, 수평·3D),
    # 그 시점, 턱이 물체에 닿은 적이 있는지, 내린 그리퍼 명령의 최솟값.
    # "엉뚱한 곳으로 갔다"와 "맞는 곳에 갔는데 못 닫았다"를 가른다.
    min_pinch_xy_mm: float = float("nan")
    min_pinch_3d_mm: float = float("nan")
    tick_at_min: int = -1
    first_contact_tick: int = -1
    gripper_cmd_min: float = float("nan")
    # The moment the gripper is commanded shut, and where the object was then.
    # `min_pinch_xy_mm` is the minimum over the whole trajectory: a lower bound on
    # the error that says nothing about whether the jaws closed there. A policy
    # that passes within 6 mm and then shuts 20 mm later fails for a different
    # reason than one that never gets close, and the two need different fixes.
    # `close_lag_ticks` = close_tick - tick_at_min; positive means it shut after
    # the closest approach had already gone by.
    # 그리퍼를 닫으라고 명령한 시점과, 그때 물체까지의 거리.
    # `min_pinch_xy_mm` 은 궤적 전체의 최솟값이라 오차의 하한일 뿐이고, 턱이 거기서
    # 닫혔는지는 말해주지 않는다. 6mm 안까지 지나갔다가 20mm 지점에서 닫는 정책과
    # 애초에 가까이 못 가는 정책은 실패 이유가 다르고 처방도 다르다.
    # `close_lag_ticks` 가 양수면 최근접 지점을 지나친 뒤에 닫았다는 뜻이다.
    close_tick: int = -1
    xy_at_close_mm: float = float("nan")
    close_lag_ticks: int = 0


def rollout(
    env: MujocoPickEnv,
    policy: Policy,
    seed: int,
    object_xy: tuple[float, float] | None = None,
) -> RolloutResult:
    """Run one episode to completion or to the tick limit.
    에피소드 하나를 완료 또는 틱 제한까지 돌린다.

    `object_xy` pins the object instead of drawing it from the seed. It exists for
    one job: checking whether a policy can reproduce the exact episode it was
    trained on. Without it, "overfit one episode and replay it" silently becomes
    "overfit one episode and test generalisation", because jitter=0 puts the object
    at the config default rather than where that episode was recorded.
    `object_xy` 는 시드에서 뽑는 대신 물체를 고정한다. 용도는 하나다 — 정책이 학습한
    바로 그 에피소드를 재현할 수 있는지 확인하는 것. 이게 없으면 "1편 과적합 후 재생"이
    조용히 "1편 과적합 후 일반화 검사"가 된다. jitter=0 은 물체를 그 에피소드가 기록된
    자리가 아니라 config 기본값에 놓기 때문이다.
    """
    obs = env.reset(seed=seed, object_xy=object_xy)
    policy.reset(seed=seed)
    obj = env.object_position()
    success = False
    ticks = 0
    min_xy, min_3d, tick_at_min = float("inf"), float("inf"), -1
    first_contact = -1
    g = env.gripper_index
    grips: list[float] = []
    xys: list[float] = []
    for _ in range(env.max_ticks):
        action = policy.act(obs)
        grips.append(float(action[g]))
        obs = env.step(action)
        ticks += 1
        xy, d3 = env.pinch_to_object_m()
        xys.append(xy)
        if xy < min_xy:
            min_xy, min_3d, tick_at_min = xy, d3, ticks
        if first_contact < 0 and env.jaw_contacts() > 0:
            first_contact = ticks
        if env.is_success():
            success = True
            break
    grip_min = min(grips) if grips else float("nan")
    close_tick, xy_at_close = closing_moment(grips, xys)
    return RolloutResult(
        seed=seed,
        success=success,
        ticks=ticks,
        lift_height_m=round(env.lift_height(), 5),
        object_xy=(round(float(obj[0]), 5), round(float(obj[1]), 5)),
        min_pinch_xy_mm=round(min_xy * 1000, 2),
        min_pinch_3d_mm=round(min_3d * 1000, 2),
        tick_at_min=tick_at_min,
        first_contact_tick=first_contact,
        gripper_cmd_min=round(grip_min, 4),
        close_tick=close_tick,
        xy_at_close_mm=round(xy_at_close * 1000, 2) if xy_at_close == xy_at_close else float("nan"),
        close_lag_ticks=(close_tick - tick_at_min) if close_tick > 0 and tick_at_min > 0 else 0,
    )


def evaluate(
    env: MujocoPickEnv,
    policy: Policy,
    seeds: list[int],
    object_xy: tuple[float, float] | None = None,
) -> tuple[float, list[RolloutResult]]:
    """Run one policy over a fixed seed list and return its success rate.
    고정된 시드 목록으로 정책 하나를 돌리고 성공률을 반환한다.

    Every policy sees the SAME seeds, so every policy sees the same object
    placements. Comparing policies across different seeds is not a comparison.
    모든 정책이 **같은** 시드를 본다. 즉 같은 물체 배치를 본다.
    다른 시드로 얻은 성공률끼리 비교하는 것은 비교가 아니다.
    """
    results = [rollout(env, policy, s, object_xy) for s in seeds]
    return sum(r.success for r in results) / len(results), results


def replay_tolerance(
    env: MujocoPickEnv, episode: Path, offsets_m: tuple[float, ...]
) -> dict[str, Any]:
    """How far the object can move before a replayed trajectory stops working.
    재생 궤적이 통하지 않게 되기까지 물체가 얼마나 움직일 수 있는가.

    This is the number that says what a policy actually has to do. If replay
    survives a 5 cm displacement, perception is not required and the task is
    mis-designed. If it dies at 1 cm, the policy must localise the object to
    better than 1 cm — which is a concrete requirement to design against.
    정책이 실제로 무엇을 해야 하는지 말해주는 수치다. 재생이 5cm 이동을 견디면
    인식이 필요 없다는 뜻이고 태스크 설계가 잘못된 것이다. 1cm 에서 죽는다면
    정책은 물체를 1cm 이내로 국소화해야 한다 — 설계에 쓸 수 있는 구체적 요구다.

    A replay that fails even at zero offset is a broken instrument, not a
    finding, so that case is measured first and reported separately.
    0 오프셋에서도 실패하는 재생은 발견이 아니라 고장난 계측기다.
    그래서 그 경우를 먼저 재고 따로 보고한다.
    """
    import json

    meta = json.loads(episode.with_suffix(".json").read_text(encoding="utf-8"))
    recorded_xy = tuple(meta["notes"]["object_init_xy"])
    policy = ReplayPolicy.from_episode(episode)

    def attempt(xy: tuple[float, float]) -> bool:
        obs = env.reset(object_xy=xy)
        policy.reset()
        for _ in range(env.max_ticks):
            obs = env.step(policy.act(obs))
            if env.is_success():
                return True
        return False

    baseline_ok = attempt(recorded_xy)
    rows: list[dict[str, Any]] = []
    for d in offsets_m:
        hits = 0
        for sign in (1.0, -1.0):
            for axis in (0, 1):
                xy = list(recorded_xy)
                xy[axis] += sign * d
                hits += int(attempt((xy[0], xy[1])))
        rows.append({"offset_m": d, "success": hits, "of": 4})
    return {
        "episode": episode.name,
        "recorded_xy": list(recorded_xy),
        "at_recorded_condition": baseline_ok,
        "rows": rows,
    }


def closing_moment(grips: list[float], xys: list[float]) -> tuple[int, float]:
    """When the gripper was commanded shut, and the distance to the object then.
    그리퍼를 닫으라고 명령한 시점과 그때 물체까지의 거리.

    A scripted policy steps the command; a learned one regresses it and wobbles.
    So the moment is not a threshold crossing but the first crossing that **holds**
    -- brushing past the threshold and coming back is not closing. The threshold
    comes from the episode's own command range, so no hardware value is hard-coded.
    스크립트 정책은 명령을 계단식으로 바꾸지만 학습 정책은 회귀 출력이라 요동친다.
    그래서 닫는 순간은 문턱 통과가 아니라 **유지되는** 첫 통과다 — 스치고 되돌아오는
    것은 닫은 것이 아니다. 문턱은 에피소드 자신의 명령 범위에서 뽑으므로 하드웨어
    값을 하드코딩하지 않는다.

    ⚠️ 이 지표가 성립하는 조건: 그리퍼 명령이 **한 번 닫히면 유지**되는 정책.
       닫았다 열었다를 반복하는 정책에는 첫 유지 구간만 잡히므로 무효다.

    에피소드가 끝나 유지 구간이 잘린 경우(성공하면 `is_success()` 로 루프가 끊긴다)
    남은 틱이 전부 문턱 아래이면 닫은 것으로 본다. 유지를 강제하면 **성공 에피소드가
    "닫은 적 없음"으로 찍힌다** — 막으려는 것은 "스치고 복귀"이지 "닫은 채 종료"가
    아니다. 다만 남은 틱이 1~2 개뿐이면 근거가 약하다는 점은 알고 쓴다.

    `(-1, nan)` means the gripper never made a closing move. That is a finding,
    not a missing value.
    `(-1, nan)` 은 닫는 동작이 아예 없었다는 뜻이다. 그것도 발견이지 결측이 아니다.
    """
    if not grips:
        return -1, float("nan")
    lo, hi = min(grips), max(grips)
    if hi - lo < GRIP_SPAN_MIN:
        return -1, float("nan")

    # 하위 문턱 아래로 내려가 CLOSE_HOLD_TICKS 만큼 **유지**된 첫 지점.
    # 스치고 되돌아오는 것은 닫은 것이 아니다.
    low = lo + CLOSE_LOW_FRAC * (hi - lo)
    n = len(grips)
    for i, gv in enumerate(grips):
        if gv > low:
            continue
        end = min(i + CLOSE_HOLD_TICKS, n)
        if all(g <= low for g in grips[i:end]) and end - i == min(CLOSE_HOLD_TICKS, n - i):
            return i + 1, xys[i]
    return -1, float("nan")


def failure_shape(results: list[RolloutResult]) -> dict[str, Any] | None:
    """Summarise how the failed attempts failed.
    실패한 시도들이 어떻게 실패했는지 요약한다."""
    fails = [r for r in results if not r.success and r.min_pinch_xy_mm == r.min_pinch_xy_mm]
    if not fails:
        return None
    xy = np.array([r.min_pinch_xy_mm for r in fails])
    contact = sum(1 for r in fails if r.first_contact_tick >= 0)
    closed = [r for r in fails if r.close_tick > 0 and r.xy_at_close_mm == r.xy_at_close_mm]
    close_block: dict[str, Any] = {"n_closed": len(closed), "never_closed": len(fails) - len(closed)}
    if closed:
        cxy = np.array([r.xy_at_close_mm for r in closed])
        lag = np.array([r.close_lag_ticks for r in closed])
        close_block.update({
            "xy_at_close_median": float(np.median(cxy)),
            "xy_at_close_q25": float(np.percentile(cxy, 25)),
            "xy_at_close_q75": float(np.percentile(cxy, 75)),
            "close_tick_median": float(np.median([r.close_tick for r in closed])),
            "lag_median": float(np.median(lag)),
            "lag_over_frac": float(np.mean(lag > CLOSE_LAG_TICKS)),
        })
    return {
        "n_fail": len(fails),
        "closing": close_block,
        "xy_median": float(np.median(xy)),
        "xy_q25": float(np.percentile(xy, 25)),
        "xy_q75": float(np.percentile(xy, 75)),
        "near_frac": float(np.mean(xy <= PRECISION_NEAR_MM)),
        "far_frac": float(np.mean(xy > PRECISION_FAR_MM)),
        "contact_frac": contact / len(fails),
        "grip_min_median": float(np.median([r.gripper_cmd_min for r in fails])),
    }


def build_policies(
    env: MujocoPickEnv, replay_from: Path | None, policy_ckpt: Path | None = None,
    device: str = "cpu", image_perturb: str | None = None,
    image_perturb_sigma: float = 96.0,
) -> list[Policy]:
    """Assemble the baseline set, skipping replay if no episode is available.
    baseline 묶음을 만든다. 재생할 에피소드가 없으면 replay 는 건너뛴다.

    A learned checkpoint joins the same list, so it is scored under exactly the
    same seeds and conditions as the baselines. A learned policy's number
    reported on its own, without the baselines beside it, means nothing.
    학습 체크포인트는 같은 목록에 들어간다. 그래야 baseline 과 **정확히 같은**
    시드·조건으로 채점된다. 학습 정책 수치를 baseline 없이 단독으로 보고하는 것은
    아무 의미가 없다.
    """
    policies: list[Policy] = [HoldPolicy(), ZeroPolicy(), ScriptedPickPolicy(env)]
    if policy_ckpt is not None:
        from policy.act import load_policy

        # `BCPolicy` defaults to cpu, and nothing recorded which device an
        # evaluation ran on. Every logged BC rollout from v2 to v6 was therefore
        # CPU inference, and the record does not say so 🟢 2026-09-07. The measured
        # difference is about one episode in a hundred -- inside every interval we
        # work with -- but "small" is not the same as "written down".
        # `BCPolicy` 기본값이 cpu 이고, 평가가 어느 장치에서 돌았는지 아무 데도
        # 기록되지 않았다. 그래서 v2~v6 의 모든 BC 롤아웃이 CPU 추론이었고 기록에는
        # 그 말이 없다 🟢 2026-09-07. 실측 차이는 100편에 1편 정도로 우리가 쓰는 모든
        # 구간 안에 들어가지만, **작다는 것과 적혀 있다는 것은 다르다.**
        # 청크 길이는 체크포인트가 정한다 — 호출자가 정하면 학습과 평가가 갈린다.
        bc = load_policy(policy_ckpt, device=device)
        print(f"학습 정책 로드: {bc.describe()}")
        if bc.meta.get("trained_on") == "random_tensors":
            print("⚠️ 이 체크포인트는 **랜덤 텐서로 학습**된 것이다. 평가 결과에 의미가 없다.")
        if image_perturb is not None:
            from policy.perturb import PerturbedObsPolicy

            # baseline 은 감싸지 않는다. hold/zero 는 이미지를 안 보고, scripted 는
            # 특권 정보를 쓰므로 교란해봐야 아무것도 말해주지 않는다.
            bc = PerturbedObsPolicy(bc, mode=image_perturb, sigma=image_perturb_sigma)
            print(f"⚠️ 추론 시점 이미지 교란: {image_perturb} "
                  f"(sigma={image_perturb_sigma}). **학습은 정상 체크포인트다.**")
        policies.append(bc)
    if replay_from is not None:
        episodes = sorted(replay_from.glob("*.npz"))
        if not episodes:
            print(f"⚠️ {replay_from} 에 에피소드가 없다. replay baseline 을 건너뛴다.")
        else:
            policies.insert(2, ReplayPolicy.from_episode(episodes[0]))
    return policies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--jitter", type=float, default=0.05,
                        help="object xy randomisation, m / 물체 xy 무작위 범위 (m)")
    parser.add_argument("--replay-from", type=Path, default=None,
                        help="dataset dir to take the replay trajectory from")
    parser.add_argument("--replay-tolerance", action="store_true",
                        help="also measure how far the object may move before replay fails")
    parser.add_argument(
        "--object-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
        help="물체를 이 좌표에 고정한다. 1편 과적합 재현 검사용 "
             "(에피소드 json 의 notes.object_init_xy 값을 넣는다)",
    )
    parser.add_argument(
        "--from-episode", type=Path, default=None,
        help="이 에피소드가 기록된 물체 위치에서 평가한다. --object-xy 를 자동으로 채운다",
    )
    parser.add_argument("--policy-ckpt", type=Path, default=None,
                        help="학습 정책 체크포인트. baseline 과 같은 조건으로 함께 채점한다")
    parser.add_argument("--render", action="store_true",
                        help="render observations; needed only for vision policies")
    # Inference-time image corruption. Training is untouched -- this answers
    # "does the trained policy look at the image", not "can it learn from images".
    # 추론 시점 이미지 교란. 학습은 건드리지 않는다 — 이것이 답하는 질문은
    # "학습된 정책이 이미지를 보는가" 이지 "이미지로 배울 수 있는가" 가 아니다.
    # `--image-noise`(학습 쪽)와 **다른 질문**이다. 0915 에 둘을 혼동했다.
    parser.add_argument("--image-perturb", type=str, default=None,
                        choices=["noise", "blackout", "freeze"],
                        help="추론 시점에 정책이 받는 이미지를 망가뜨린다")
    parser.add_argument("--image-perturb-sigma", type=float, default=96.0,
                        help="--image-perturb noise 의 표준편차. 학습 쪽 기본값과 같다")
    parser.add_argument("--policy-device", type=str, default="cpu",
                        help="학습 정책 추론 장치. 기본 cpu — v2~v6 이력이 전부 cpu 라 "
                             "비교선을 유지한다. 바꾸면 conditions 에 기록되므로 "
                             "표류 감지기가 잡는다")
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg: dict[str, Any] = load_config()
    seeds = [args.seed_base + i for i in range(args.episodes)]

    object_xy: tuple[float, float] | None = None
    if args.from_episode is not None:
        import json as _json

        meta = _json.loads(args.from_episode.with_suffix(".json").read_text(encoding="utf-8"))
        xy = meta["notes"]["object_init_xy"]
        object_xy = (float(xy[0]), float(xy[1]))
        print(f"물체를 {args.from_episode.name} 이 기록된 위치 "
              f"({object_xy[0]:+.4f}, {object_xy[1]:+.4f}) 에 고정한다")
    elif args.object_xy is not None:
        object_xy = (float(args.object_xy[0]), float(args.object_xy[1]))
        print(f"물체를 ({object_xy[0]:+.4f}, {object_xy[1]:+.4f}) 에 고정한다")

    if object_xy is not None and args.episodes > 1:
        print(f"⚠️ 물체가 고정돼 있어 {args.episodes}회가 전부 같은 조건이다. "
              "시뮬은 결정적이므로 성공/실패도 같다 — 표본이 아니다")

    print("게이트 기준 (결과 확인 전 확정):")
    for key, text in GATES.items():
        print(f"  [{key}] {text}")
    print()

    rates: dict[str, float] = {}
    shapes: dict[str, dict[str, Any]] = {}
    detail: dict[str, list[dict[str, Any]]] = {}
    privileged: dict[str, bool] = {}
    action_space: str | None = None

    with MujocoPickEnv(cfg, render=args.render, object_jitter_m=args.jitter) as env:
        for policy in build_policies(env, args.replay_from, args.policy_ckpt,
                                     device=args.policy_device,
                                     image_perturb=args.image_perturb,
                                     image_perturb_sigma=args.image_perturb_sigma):
            rate, results = evaluate(env, policy, seeds, object_xy)
            if policy.name == "bc":
                action_space = getattr(policy, "action_space", "joint_absolute")
            rates[policy.name] = rate
            privileged[policy.name] = policy.uses_privileged_state
            detail[policy.name] = [r.__dict__ for r in results]
            mark = " (특권정보 사용 — 실물 배포 불가)" if policy.uses_privileged_state else ""
            print(
                f"{policy.name:10s} {sum(r.success for r in results):3d}/{len(results)} "
                f"= {rate * 100:5.1f}%   평균 상승 {np.mean([r.lift_height_m for r in results]) * 100:5.2f}cm{mark}"
            )
            shape = failure_shape(results)
            if shape is not None and policy.name not in ("hold", "zero"):
                shapes[policy.name] = shape
                print(
                    f"           실패 {shape['n_fail']}건의 파지점-물체 최소 수평거리: "
                    f"중앙값 {shape['xy_median']:.1f}mm (Q1 {shape['xy_q25']:.1f} / Q3 {shape['xy_q75']:.1f}) · "
                    f"≤{PRECISION_NEAR_MM:.0f}mm {shape['near_frac'] * 100:.0f}% · "
                    f">{PRECISION_FAR_MM:.0f}mm {shape['far_frac'] * 100:.0f}% · "
                    f"턱 접촉 {shape['contact_frac'] * 100:.0f}% · 그리퍼 명령 최솟값 중앙 {shape['grip_min_median']:+.3f}"
                )
                cb = shape["closing"]
                if cb.get("xy_at_close_median") is not None:
                    print(
                        f"           닫는 순간 거리 중앙 {cb['xy_at_close_median']:.1f}mm "
                        f"(Q1 {cb['xy_at_close_q25']:.1f} / Q3 {cb['xy_at_close_q75']:.1f}) · "
                        f"닫은 틱 중앙 {cb['close_tick_median']:.0f} · "
                        f"최근접 대비 지연 중앙 {cb['lag_median']:+.0f}틱 "
                        f"(>{CLOSE_LAG_TICKS}틱 {cb['lag_over_frac'] * 100:.0f}%)"
                        + (f" · 닫지 않음 {cb['never_closed']}건" if cb["never_closed"] else "")
                    )
                else:
                    print(f"           ⚠️ 실패 {cb['never_closed']}건 전부 닫는 동작이 없었다 "
                          f"(그리퍼 명령 진폭 < {GRIP_SPAN_MIN})")

    where = (f"물체 고정 ({object_xy[0]:+.4f}, {object_xy[1]:+.4f})"
             if object_xy is not None else f"물체 xy ±{args.jitter * 1000:.0f}mm")
    print(f"\n조건: {where}, 시드 {seeds[0]}~{seeds[-1]}, "
          f"모든 정책이 동일 시드, 틱 제한 {MujocoPickEnv(cfg, render=False).max_ticks}")

    tolerance: dict[str, Any] | None = None
    if args.replay_tolerance and args.replay_from is not None:
        episodes = sorted(args.replay_from.glob("*.npz"))
        if episodes:
            with MujocoPickEnv(cfg, render=False, object_jitter_m=args.jitter) as env:
                tolerance = replay_tolerance(
                    env, episodes[0], (0.005, 0.010, 0.015, 0.020, 0.030, 0.050)
                )
            print(f"\nreplay 위치 허용오차 — 원본 {tolerance['episode']}, "
                  f"기록 위치 {tolerance['recorded_xy']}")
            state = "성공" if tolerance["at_recorded_condition"] else "실패 — 계측기가 고장난 것이므로 아래 수치는 무효"
            print(f"  기록된 바로 그 위치에서: {state}")
            for row in tolerance["rows"]:
                print(f"  물체를 ±{row['offset_m'] * 1000:4.0f}mm 이동 (4방향)  {row['success']}/4")

    print("\n게이트 판정:")
    if "replay" in rates:
        ok = rates["replay"] < 0.30
        print(f"  [task_validity] replay {rates['replay'] * 100:.1f}% < 30% → "
              f"{'통과 — 이 태스크는 관측을 봐야 풀린다' if ok else '실패 — 고정 궤적으로 풀린다. 태스크 재설계 필요'}")
    if "scripted" in rates:
        ok = rates["scripted"] >= 0.80
        print(f"  [ceiling]       scripted {rates['scripted'] * 100:.1f}% >= 80% → "
              f"{'통과' if ok else '실패 — 태스크/씬 문제이지 정책 문제가 아니다'}")
    floor = max(rates.get("hold", 0.0), rates.get("zero", 0.0))
    threshold = floor + 0.20
    if "bc" in shapes:
        sh = shapes["bc"]
        if sh["xy_median"] <= PRECISION_NEAR_MM:
            read = ("가까이는 간다 — 접근은 되고 그 뒤가 안 된다. "
                    "원인은 이 값으로 정해지지 않는다")
        elif sh["xy_median"] > PRECISION_FAR_MM:
            read = "접근 자체가 틀린다 — 위치 추정을 먼저 본다"
        else:
            read = "판정 유보 — 중간 영역"
        print(f"  [failure_shape] bc 실패 최소거리 중앙값 {sh['xy_median']:.1f}mm → {read}")
        cb = sh["closing"]
        if cb.get("xy_at_close_median") is not None:
            cm, lg = cb["xy_at_close_median"], cb["lag_median"]
            drift = cm - sh["xy_median"]
            if cm > CLOSE_BAD_MM:
                cread = "실측 허용오차 밖이다 (±15mm 는 1/4 만 성공)"
            elif cm <= CLOSE_OK_MM:
                cread = "실측 허용오차 안이다 (±5mm 4/4) — 위치는 충분했다"
            else:
                cread = "중간 영역 (±10mm 3/4)"
            print(f"  [closing_moment] 닫는 순간 {cm:.1f}mm — {cread}")
            print(f"                  최근접({sh['xy_median']:.1f}mm) 대비 {drift:+.1f}mm · "
                  f"지연 {lg:+.0f}틱 (지연은 판정 근거가 아니다 — scripted 는 +30틱에 84% 🟢)")
            print("                  ⚠️ 이 값은 실패 **원인**을 정하지 않는다. "
                  "2026-09-07 실측 🟢: 닫는 순간 xy 2.7mm 로 제대로 잡고도 "
                  "들어올리지 못해 실패한 사례가 있다. 원인은 폐루프 궤적을 봐야 한다 "
                  "(tools/trace_execution.py)")
        elif cb["never_closed"]:
            print(f"  [closing_moment] 실패 {cb['never_closed']}건이 닫는 동작 자체를 하지 않았다 "
                  "→ 그리퍼 출력이 죽어 있다")
    if "bc" in rates:
        ok = rates["bc"] > threshold
        print(f"  [floor/chance]  bc {rates['bc'] * 100:.1f}% > {threshold * 100:.1f}% → "
              f"{'통과' if ok else '**실패 — baseline 대비 의미 있는 차이가 아니다**'}")
        if "scripted" in rates and rates["scripted"] > 0:
            gap = rates["scripted"] - rates["bc"]
            print(f"                  상한(scripted) 대비 {gap * 100:.1f}%p 아래 — "
                  f"이미지로 위치를 추정하는 데 드는 비용이다")
    else:
        print(f"  [floor/chance]  학습 정책은 {threshold * 100:.1f}% 를 넘어야 의미가 있다 "
              f"(hold {rates.get('hold', 0) * 100:.1f}%, zero {rates.get('zero', 0) * 100:.1f}%)")

    if args.log:
        rec = log_run(
            experiment="rollout_baselines",
            author=args.author,
            issue="S15P21A103-60",
            conditions={
                "episodes": args.episodes,
                "seeds": [seeds[0], seeds[-1]],
                "jitter_m": args.jitter,
                "object_xy_fixed": list(object_xy) if object_xy else None,
                "render": args.render,
                "config_sha": file_digest(DEFAULT_CONFIG),
                # Without these a rollout number cannot be attributed to a
                # checkpoint. "bc scored 25%" is not a finding if nobody can say
                # which bc, trained in which action space.
                # 이게 없으면 롤아웃 수치를 체크포인트에 귀속시킬 수 없다.
                # 어느 bc 인지, 어느 행동 공간으로 학습된 것인지 말할 수 없으면
                # "bc 가 25% 였다"는 발견이 아니다.
                "policy_ckpt": str(args.policy_ckpt) if args.policy_ckpt else None,
                "policy_action_space": action_space,
                # ⚠️ 교란 조건이 여기 없으면 교란 실행과 정상 실행이 **같은
                # policy_ckpt 로 조회돼 서로를 집어간다.** 0915 오전에 `train_bc`
                # 기록에서 정확히 같은 버그를 겪었다 (8잡이 서로의 val_loss 를
                # 가져갔다). 조건을 바꿨으면 조건에 적는다.
                "image_perturb": args.image_perturb,
                "image_perturb_sigma": (
                    args.image_perturb_sigma if args.image_perturb else None
                ),
                "gates": GATES,
                "policy_device": args.policy_device,
                "reading_rules": {
                    "precision_near_mm": PRECISION_NEAR_MM, "precision_far_mm": PRECISION_FAR_MM,
                    "close_ok_mm": CLOSE_OK_MM, "close_bad_mm": CLOSE_BAD_MM,
                    "close_lag_ticks": CLOSE_LAG_TICKS,
                },
                "metric": "롤아웃 성공률 (validation loss 아님)",
            },
            result={
                "success_rates": rates,
                "failure_shape": shapes,
                "uses_privileged_state": privileged,
                "replay_tolerance": tolerance,
                "detail": detail,
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")


if __name__ == "__main__":
    main()
