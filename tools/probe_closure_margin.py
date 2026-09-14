"""Required squeeze margin: grasp success vs commanded gap relative to object width.
조임 여유 — 명령 개구가 물체 폭보다 얼마나 좁아야 파지가 성립하는가.

왜 f(폐쇄 비율)가 아니라 mm 인가 / Why millimetres, not a closure fraction.

`probe_gripper_closure.py`(0912)는 폐쇄량을 개방→폐쇄 구간의 **비율 f** 로 잘랐다.
f=0.80 에서 이미 0% 가 나왔는데, config 의 `grasp.gap_curve` 로 환산해 보면 이유가
신비롭지 않다 🟢:

    f=1.00 -> 관절 0.000 rad -> 명령 개구 14.7mm  (20mm 큐브를 5.3mm 조인다)
    f=0.80 -> 관절 0.120 rad -> 명령 개구 21.4mm  (큐브보다 1.4mm 넓다. 닿지 않는다)
    f=0.35 -> 관절 0.390 rad -> 명령 개구 36.3mm  (0911 실수집분의 폐쇄량)

즉 절벽의 위치는 **물체 폭**이 정한다. f 는 물체가 바뀌면 뜻이 달라지는 축이라
수집 프로토콜에 쓸 수 없다. `명령 개구 - 물체 폭` 은 폭이 바뀌어도 같은 물리량이고,
그대로 문장이 된다 — "폭 W 부품은 W 보다 X mm 좁게까지 쥔다".

교차 확인 🟢: f=0.35 의 환산값 36.3mm 는 마커 영상 실측의 평탄 구간 38~40mm
(`MEASURE_gripper_crosscheck_0912.md`)와 몇 mm 안에서 만난다. 정규화 행동공간에서
나온 값과 영상에서 나온 값은 독립 경로다.

무엇을 답하고 무엇을 답하지 않는가.

  답한다   이 시뮬 태스크에서 폭 W 물체를 쥐려면 명령 개구가 W 보다 몇 mm 좁아야 하는가
  답한다   0912 절벽이 **물체 폭** 때문인가 **시뮬 물리 설정**(마찰·페이로드) 때문인가
           -- 폭을 바꿨을 때 필요 여유가 같이 움직이면 폭, 안 움직이면 물리 설정이다
  답 안 한다  실물에서 같은 여유가 필요한가. 시뮬 성공률은 sim2real 갭의 하한이다

하드코딩하지 않는다. 관절 범위·`gap_curve`·물체 크기 전부 config 에서 읽는다.
로봇이 변형 예정이므로 상수를 여기 박으면 조용히 틀린다.

사용:
    # [서버]
    python tools/probe_closure_margin.py --episodes 100 --out out/margin.json --log
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

# tools/ 관례. PYTHONPATH 없이 불러도 패키지를 찾게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402  — numpy/torch 앞에 와야 한다

_claim_name = "probe_closure_margin"
if "--claim-name" in sys.argv:
    _i = sys.argv.index("--claim-name")
    if _i + 1 >= len(sys.argv):
        raise ValueError("--claim-name requires a value")
    _claim_name = sys.argv[_i + 1]
runtime_limits.claim(_claim_name)
runtime_limits.torch_threads()

import numpy as np  # noqa: E402

from eval.rollout import rollout  # noqa: E402
from eval.stats import wilson_ci  # noqa: E402
from policy.baselines import ScriptedPickPolicy  # noqa: E402
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from sim.mujoco.env import MujocoPickEnv  # noqa: E402
from tracking.exp_log import log_run  # noqa: E402


OBJECT_WIDTHS_MM: tuple[float, ...] = (15.0, 20.0, 25.0)
"""타겟 도메인 규격 1.5~2.5cm 의 양끝과 가운데. 20mm 는 기존 baseline 과 같은 큐브다.
결과를 보고 목록을 늘리지 않는다."""

MARGINS_MM: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0)
"""명령 개구 - 물체 폭. 음수 = 물체보다 좁게 명령(조인다), 양수 = 닿지 않는다.
0912 실측으로 f=1.00 이 20mm 큐브에서 -5.3mm 였고 +1.4mm 에서 0% 였으므로
절벽은 이 구간 안에 있다."""


def gap_curve_to_rad(gap_curve: list[list[float]], gap_cm: float) -> float:
    """Invert the config's joint-angle -> pad-gap curve.
    config 의 관절각 -> 패드 개구 곡선을 뒤집는다.

    `gap_curve` 는 (rad, gap_cm) 오름차순 점열이다. 곡선 밖 값은 외삽하지 않고
    막는다 — 물리적으로 낼 수 없는 개구를 조용히 반환하면 그 조건의 수치가 거짓이 된다.
    """
    rads = [float(p[0]) for p in gap_curve]
    gaps = [float(p[1]) for p in gap_curve]
    if not all(b > a for a, b in zip(gaps, gaps[1:])):
        raise ValueError("gap_curve 의 개구가 단조증가가 아니다 — 역함수를 정의할 수 없다")
    if not (gaps[0] <= gap_cm <= gaps[-1]):
        raise ValueError(
            f"개구 {gap_cm * 10:.1f}mm 는 이 그리퍼의 물리 범위 "
            f"{gaps[0] * 10:.1f}~{gaps[-1] * 10:.1f}mm 밖이다"
        )
    return float(np.interp(gap_cm, gaps, rads))


def rad_to_norm(rad: float, lo: float, hi: float) -> float:
    """configs/so101.yaml 의 normalization.formula 그대로."""
    return 2.0 * (rad - lo) / (hi - lo) - 1.0


class GapCeiling:
    """Clamp the gripper command so the commanded pad gap never goes below a floor.
    명령 개구가 바닥값보다 좁아지지 못하게 그리퍼 채널만 막는다.

    팔 5채널은 건드리지 않는다. 실수집분에서 관측된 모양이 그것이다 —
    팔은 맞는 자리로 가고 턱만 덜 닫힌다.
    """

    uses_privileged_state = True

    def __init__(self, base, gripper_index: int, floor_norm: float, label: str) -> None:
        self.base = base
        self.g = int(gripper_index)
        self.floor = float(floor_norm)
        self.name = label

    def reset(self, seed: int | None = None) -> None:
        self.base.reset(seed)

    def act(self, obs) -> np.ndarray:
        action = np.array(self.base.act(obs), dtype=np.float64, copy=True)
        if action[self.g] < self.floor:
            action[self.g] = self.floor
        return action


def _summarise(results) -> dict:
    n = len(results)
    ok = sum(1 for r in results if r.success)
    lo, hi = wilson_ci(ok, n)
    contact = sum(1 for r in results if r.first_contact_tick > 0)
    xy = [r.min_pinch_xy_mm for r in results if r.min_pinch_xy_mm == r.min_pinch_xy_mm]
    return {
        "n": n,
        "success": ok,
        "rate": round(ok / n, 4) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
        "contact_rate": round(contact / n, 4) if n else 0.0,
        "min_pinch_xy_mm_median": round(float(np.median(xy)), 2) if xy else None,
        "success_seeds": [r.seed for r in results if r.success],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--episodes", type=int, default=100,
                    help="시드 수. n=20 은 게이트 판정이 안 된다 (구간 반폭 약 ±18%%p)")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--jitter", type=float, default=0.05,
                    help="물체 xy 무작위 범위 (m). 기존 baseline 과 같은 0.05")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--claim-name", default="probe_closure_margin")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", action="store_true", help="EXP_LOG.jsonl 에 append")
    args = ap.parse_args()

    base_cfg = load_config(args.config) if args.config else load_config(DEFAULT_CONFIG)
    seeds = [args.seed_base + i for i in range(args.episodes)]

    gripper = next(j for j in base_cfg["joints"] if j["name"] == "gripper")
    lo_rad, hi_rad = (float(x) for x in gripper["range_rad"])
    curve = base_cfg["grasp"]["gap_curve"]
    gap_min_mm, gap_max_mm = curve[0][1] * 10.0, curve[-1][1] * 10.0

    print(f"조건: 물체 xy ±{args.jitter * 1000:.0f}mm · 시드 {seeds[0]}~{seeds[-1]} "
          f"(n={len(seeds)}) · 모든 조건 동일 시드 · scripted 천장 · 학습 없음")
    print(f"그리퍼 물리 개구 범위 (config gap_curve): {gap_min_mm:.1f}~{gap_max_mm:.1f}mm")
    print(f"사전등록: 물체 폭 {OBJECT_WIDTHS_MM} mm × 조임 여유 {MARGINS_MM} mm")
    print("판정 기준은 결과를 보기 전에 박는다 —")
    print("  세 폭에서 필요 여유가 **같이 움직이면** 원인은 물체 폭이다")
    print("  세 폭에서 필요 여유가 **거의 같으면** 원인은 시뮬 물리 설정(마찰·페이로드)이고,")
    print("    그 경우 이 결과를 실물 재수집 근거로 쓰지 않는다")
    print()

    results: dict[str, dict] = {}
    instrument_ok = True

    for width_mm in OBJECT_WIDTHS_MM:
        half = width_mm / 2000.0                      # mm -> m, 반치수
        cfg = copy.deepcopy(base_cfg)
        cfg["task"]["object"]["half_size_m"] = [half, half, half]
        # 기본 config 는 반치수 0.01 에 z=0.011 — 바닥에서 1mm 띄운다. 그 규칙을 유지한다.
        x, y = cfg["task"]["object"]["init_pos"][:2]
        cfg["task"]["object"]["init_pos"] = [x, y, half + 0.001]

        # 천장 정책의 폐쇄 명령도 폭에 맞춰 옮긴다.
        # 기본 config 의 close_cmd 는 0.06 rad = 개구 18.0mm 로 **고정**이라, 그대로 두면
        # 15mm 물체는 천장 정책조차 못 쥔다(명령이 물체보다 3mm 넓다). 그러면 그 폭의
        # 행 전체가 "여유 부족"이 아니라 "기준선 자체가 0%"가 되어 아무것도 못 읽는다.
        # 20mm 기준선의 조임량(18.0mm 명령 vs 20mm 물체 = 2mm)을 폭마다 보존한다.
        baseline_squeeze_mm = 2.0
        cfg["grasp"]["close_cmd"] = gap_curve_to_rad(
            curve, (width_mm - baseline_squeeze_mm) / 10.0
        )

        print(f"── 물체 폭 {width_mm:.0f}mm (반치수 {half * 1000:.1f}mm) · "
              f"천장 폐쇄명령 {width_mm - baseline_squeeze_mm:.1f}mm "
              f"({cfg['grasp']['close_cmd']:+.4f} rad)"
              f"{'  ← 기존 baseline' if abs(width_mm - 20.0) < 1e-9 else ''}")

        with MujocoPickEnv(cfg, render=False, object_jitter_m=args.jitter) as env:
            g = env.gripper_index
            unclamped = [rollout(env, ScriptedPickPolicy(env), s) for s in seeds]
            u = _summarise(unclamped)
            results[f"w{width_mm:.0f}_unclamped"] = u
            print(f"   상한 없음   {u['success']:>3}/{u['n']} = {u['rate']:>6.1%} "
                  f"[{u['ci95'][0]:.2f}, {u['ci95'][1]:.2f}]  접촉 {u['contact_rate']:.0%}")

            if u["success"] == 0:
                print("   ⚠️ 이 폭에서는 상한 없이도 0% 다 — 조임 여유가 아니라 "
                      "다른 요인(도달·접근)이 먼저 막는다. 아래 조건은 해석하지 않는다")

            for margin in MARGINS_MM:
                target_gap_mm = width_mm + margin
                label = f"w{width_mm:.0f}_m{margin:+.0f}"
                if not (gap_min_mm <= target_gap_mm <= gap_max_mm):
                    results[label] = {"skipped": "그리퍼 물리 범위 밖",
                                      "target_gap_mm": target_gap_mm}
                    print(f"   여유 {margin:+.0f}mm (개구 {target_gap_mm:.1f}mm)  "
                          f"건너뜀 — 물리 범위 밖")
                    continue
                floor_rad = gap_curve_to_rad(curve, target_gap_mm / 10.0)
                floor_norm = rad_to_norm(floor_rad, lo_rad, hi_rad)
                pol = GapCeiling(ScriptedPickPolicy(env), g, floor_norm, label)
                r = _summarise([rollout(env, pol, s) for s in seeds])
                r.update(object_width_mm=width_mm, margin_mm=margin,
                         target_gap_mm=round(target_gap_mm, 2),
                         floor_rad=round(floor_rad, 5),
                         floor_norm=round(floor_norm, 5))
                results[label] = r
                print(f"   여유 {margin:+.0f}mm (개구 {target_gap_mm:>4.1f}mm)  "
                      f"{r['success']:>3}/{r['n']} = {r['rate']:>6.1%} "
                      f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]  "
                      f"접촉 {r['contact_rate']:.0%}")
        print()

    # --- 계측기 검정 -----------------------------------------------------------
    # 각 폭에서 "가장 좁게 명령한 조건"이 상한 없음보다 나빠지면 래퍼가 고장난 것이다.
    # 상한 없음은 config 의 close_cmd 까지 내려가므로, 그보다 더 좁은 바닥은 아무것도
    # 막지 않아야 한다.
    print("계측기 검정 — 가장 좁은 여유가 상한 없음과 같은 성공 집합을 내야 한다")
    for width_mm in OBJECT_WIDTHS_MM:
        u = results.get(f"w{width_mm:.0f}_unclamped")
        tight = results.get(f"w{width_mm:.0f}_m{MARGINS_MM[0]:+.0f}")
        if not u or not tight or "success_seeds" not in tight:
            print(f"  폭 {width_mm:.0f}mm: 판정 불가 (조건 누락)")
            continue
        a, b = set(u["success_seeds"]), set(tight["success_seeds"])
        ok = a == b
        instrument_ok &= ok
        print(f"  폭 {width_mm:.0f}mm: {'OK' if ok else '✗ 불일치'} "
              f"(상한없음 {len(a)}개 · 여유{MARGINS_MM[0]:+.0f}mm {len(b)}개)")
    if not instrument_ok:
        print("  ✗ 한 폭 이상에서 불일치다. **이 실행의 수치는 무효다.**")

    print()
    print("⚠️ 전부 시뮬이다. 시뮬 성공률은 sim2real 갭의 하한이다 — "
          "여기서 무너지면 실물에서도 무너지지만, 버틴다고 실물을 보장하지 않는다.")

    payload = {
        "instrument_ok": instrument_ok,
        "object_widths_mm": list(OBJECT_WIDTHS_MM),
        "margins_mm": list(MARGINS_MM),
        "gripper_gap_range_mm": [round(gap_min_mm, 2), round(gap_max_mm, 2)],
        "gripper_range_rad": [lo_rad, hi_rad],
        "baseline_squeeze_mm": 2.0,
        "conditions": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"결과: {args.out}")

    if args.log:
        log_run(
            experiment="closure_margin_sweep",
            author="김준태(트랙B)",
            issue="S15P21A103-34",
            conditions={
                "episodes": args.episodes,
                "seed_base": args.seed_base,
                "jitter_m": args.jitter,
                "object_widths_mm": list(OBJECT_WIDTHS_MM),
                "margins_mm": list(MARGINS_MM),
                "policy": "scripted (천장)",
                "render": False,
                "prereg": "docs/PREREG_closure_margin_0913.md",
            },
            result=payload,
        )

    return 0 if instrument_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
