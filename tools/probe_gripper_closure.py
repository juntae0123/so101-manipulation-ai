"""Cost of an under-closing gripper: scripted grasp under a closure ceiling.
폐쇄량 상한을 건 스크립트 파지 — 덜 닫히는 그리퍼가 치르는 비용을 잰다.

왜 있는가 / Why this exists.

2026-09-11 실수집 71편에서 **그리퍼가 끝까지 닫히지 않는다** 🟢.
  마커 실측: gap < 30mm 가 5,759프레임 중 5개(0.09%), 편별 최소 gap 중앙 37.26mm
  계약 변환본: `action[:,5]` 최솟값 -0.2922, 닫힘은 -1.0 → 개방→닫힘의 약 35% 지점
  (`MEASURE_gripper_crosscheck_0912.md` · `MEASURE_real_action_floor_0912.md`)

이 상태로 HW·PM 에 "재수집이 필요하다" 고 말하고 있는데, 그것은 **논증이지 측정이
아니다.** 절대원칙 2 — 대책보다 계측기를 먼저 낸다. 시뮬에서는 잴 수 있다:
천장 정책(scripted)의 그리퍼 명령에 폐쇄 상한을 걸고 성공률이 어떻게 무너지는지 본다.

무엇을 답하고 무엇을 답하지 않는가.

  답한다   폐쇄량이 f 까지만 갈 때 **이 태스크·이 물체**에서 파지가 살아남는가
  답 안 한다  실물에서 같은 f 가 같은 결과를 내는가. 시뮬 성공률은 sim2real 갭의
              하한이다. 여기서 무너지면 실물에서도 무너지지만 역은 성립하지 않는다

계측기 검정 / Instrument check.

f=1.00 은 상한을 걸지 않은 것과 **같은 성공 집합**을 내야 한다. 다르면 래퍼가
고장난 것이고, 아래 수치는 전부 무효다. 그래서 f=1.00 을 조건에 넣고 pass 0 과
시드 단위로 대조한다. 고장난 계측기는 스스로 알리지 않는다.

폐쇄량 기준값은 **하드코딩하지 않는다.** pass 0 에서 scripted 가 실제로 낸 명령의
최대(개방)와 최소(폐쇄)를 관측해 그 사이를 f 로 자른다. 관절 범위가 configs/ 에
있고 로봇이 변형 예정이므로, 상수를 여기 박으면 로봇이 바뀔 때 조용히 틀린다.

사용:
    # [서버]
    python tools/probe_gripper_closure.py --episodes 100 --out out/closure.json --log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# tools/ 관례. PYTHONPATH 없이 불러도 패키지를 찾게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402  — numpy/torch 앞에 와야 한다

_claim_name = "probe_gripper_closure"
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


CLOSURE_FRACTIONS: tuple[float, ...] = (1.00, 0.80, 0.60, 0.50, 0.40, 0.35, 0.20)
"""재는 폐쇄량. 0.35 가 실수집 관측치다 — 나머지는 그 주변의 기울기를 보기 위한 것이다.
결과를 보고 목록을 늘리지 않는다. 늘리면 그것은 다중비교다."""

REAL_OBSERVED_FRACTION = 0.35
"""0911 실수집분의 폐쇄량 🟢 `MEASURE_real_action_floor_0912.md` 4절.
정규화 -0.2922 (관측 최솟값) 이 -1.0(닫힘) 까지 갈 길의 약 35% 지점이다."""


class ClosureCeiling:
    """Wrap a policy so its gripper command can never go below a floor.
    그리퍼 명령이 바닥값 아래로 못 내려가게 감싼다.

    나머지 5채널은 건드리지 않는다. 팔은 똑같이 움직이고 **턱만 덜 닫힌다** —
    이것이 실수집에서 관측된 모양이다. 팔까지 바꾸면 무엇이 원인인지 못 가른다.
    """

    uses_privileged_state = True

    def __init__(self, base, gripper_index: int, floor: float, label: str) -> None:
        self.base = base
        self.g = int(gripper_index)
        self.floor = float(floor)
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
    gmin = [r.gripper_cmd_min for r in results if r.gripper_cmd_min == r.gripper_cmd_min]
    return {
        "n": n,
        "success": ok,
        "rate": round(ok / n, 4) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
        "contact_rate": round(contact / n, 4) if n else 0.0,
        "min_pinch_xy_mm_median": round(float(np.median(xy)), 2) if xy else None,
        "gripper_cmd_min_observed": round(float(np.min(gmin)), 4) if gmin else None,
        "success_seeds": [r.seed for r in results if r.success],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--episodes", type=int, default=100,
                    help="시드 수. n=20 은 게이트 판정이 안 된다 (구간 반폭 약 ±18%p)")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--jitter", type=float, default=0.05,
                    help="물체 xy 무작위 범위 (m). 기존 baseline 과 같은 0.05 를 쓴다")
    ap.add_argument("--config", type=Path, default=None,
                    help="씬 config. 생략하면 DEFAULT_CONFIG")
    ap.add_argument("--claim-name", default="probe_gripper_closure")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", action="store_true", help="EXP_LOG.jsonl 에 append")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config(DEFAULT_CONFIG)
    seeds = [args.seed_base + i for i in range(args.episodes)]

    print(f"조건: 물체 xy ±{args.jitter * 1000:.0f}mm · 시드 {seeds[0]}~{seeds[-1]} "
          f"(n={len(seeds)}) · 모든 조건이 동일 시드")
    print(f"사전등록: 재는 폐쇄량 {CLOSURE_FRACTIONS} · 실수집 관측치 {REAL_OBSERVED_FRACTION}")
    print("판정 기준은 결과를 보기 전에 박는다 —")
    print("  f=0.35 성공률 < 10%  → 재수집 필수 판정의 근거가 된다")
    print("  f=0.35 성공률 >= 50% → 내 에스컬레이션이 과했던 것으로 기록한다")
    print("  그 사이                → 판정 보류. 실물 확인(대상물 실측 폭)이 먼저다")
    print()

    conditions: dict[str, dict] = {}

    with MujocoPickEnv(cfg, render=False, object_jitter_m=args.jitter) as env:
        g = env.gripper_index

        # --- pass 0: 상한 없는 천장. 여기서 개방·폐쇄 명령 범위를 관측한다 ---
        base = ScriptedPickPolicy(env)
        raw = [rollout(env, base, s) for s in seeds]
        conditions["unclamped"] = _summarise(raw)
        close_cmd = min(r.gripper_cmd_min for r in raw)

        # 개방값은 궤적 최대가 필요하다. rollout 은 최솟값만 남기므로 한 편만 다시
        # 돌려 명령을 직접 모은다 — 시드 하나면 충분하다 (스크립트는 같은 개방값에서 출발한다).
        probe = ScriptedPickPolicy(env)
        obs = env.reset(seed=seeds[0])
        probe.reset(seed=seeds[0])
        cmds = []
        for _ in range(env.max_ticks):
            a = probe.act(obs)
            cmds.append(float(a[g]))
            obs = env.step(a)
            if env.is_success():
                break
        open_cmd = max(cmds)

        travel = open_cmd - close_cmd
        print(f"관측된 그리퍼 명령 범위: 개방 {open_cmd:+.4f} → 폐쇄 {close_cmd:+.4f} "
              f"(이동량 {travel:.4f})")
        if travel <= 1e-6:
            print("✗ 스크립트 정책이 그리퍼를 움직이지 않는다. 상한을 걸 대상이 없다.",
                  file=sys.stderr)
            return 2
        print(f"천장(상한 없음): {conditions['unclamped']['success']}/{len(seeds)} "
              f"= {conditions['unclamped']['rate']:.1%}")
        print()

        # --- 폐쇄량별 ---
        for f in CLOSURE_FRACTIONS:
            floor = open_cmd - f * travel
            label = f"f{f:.2f}"
            pol = ClosureCeiling(ScriptedPickPolicy(env), g, floor, label)
            res = [rollout(env, pol, s) for s in seeds]
            s = _summarise(res)
            s["closure_fraction"] = f
            s["gripper_floor_cmd"] = round(floor, 4)
            conditions[label] = s
            mark = "  ← 실수집 관측치" if abs(f - REAL_OBSERVED_FRACTION) < 1e-9 else ""
            print(f"  f={f:.2f}  바닥 {floor:+.4f}  "
                  f"{s['success']:>3}/{s['n']} = {s['rate']:>6.1%} "
                  f"[{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]  "
                  f"접촉 {s['contact_rate']:.0%}  "
                  f"최근접 중앙 {s['min_pinch_xy_mm_median']}mm{mark}")

    # --- 계측기 검정 ---
    a = set(conditions["unclamped"]["success_seeds"])
    b = set(conditions["f1.00"]["success_seeds"])
    instrument_ok = a == b
    print()
    if instrument_ok:
        print(f"계측기 검정 OK — f=1.00 이 상한 없음과 성공 시드 집합 일치 ({len(a)}개)")
    else:
        print(f"✗ 계측기 검정 실패 — f=1.00 이 상한 없음과 다르다 "
              f"(상한없음만 {sorted(a - b)} · f1.00만 {sorted(b - a)})")
        print("  래퍼가 그리퍼 외 채널을 건드렸거나 바닥값 계산이 틀렸다.")
        print("  **아래 수치는 전부 무효다.**")

    real = conditions[f"f{REAL_OBSERVED_FRACTION:.2f}"]
    print()
    print(f"실수집 관측 폐쇄량 f={REAL_OBSERVED_FRACTION:.2f} 에서 "
          f"{real['success']}/{real['n']} = {real['rate']:.1%} "
          f"[{real['ci95'][0]:.2f}, {real['ci95'][1]:.2f}]")
    if not instrument_ok:
        verdict = "무효 — 계측기 검정 실패"
    elif real["rate"] < 0.10:
        verdict = "재수집 필수 판정의 근거가 된다 (사전등록 기준 <10%)"
    elif real["rate"] >= 0.50:
        verdict = "에스컬레이션이 과했다 (사전등록 기준 >=50%). 그대로 기록한다"
    else:
        verdict = "판정 보류 (사전등록 기준 10~50%). 대상물 실측 폭 확인이 먼저다"
    print(f"판정: {verdict}")
    print()
    print("⚠️ 전부 시뮬이다. 시뮬 성공률은 sim2real 갭의 **하한**이다 — "
          "여기서 무너지면 실물에서도 무너지지만, 버틴다고 실물을 보장하지 않는다.")

    payload = {
        "instrument_ok": instrument_ok,
        "open_cmd": round(open_cmd, 4),
        "close_cmd": round(close_cmd, 4),
        "real_observed_fraction": REAL_OBSERVED_FRACTION,
        "verdict": verdict,
        "conditions": conditions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"결과: {args.out}")

    if args.log:
        log_run(
            experiment="gripper_closure_ceiling",
            author="김준태(트랙B)",
            issue="S15P21A103-34",
            conditions={
                "episodes": args.episodes,
                "seed_base": args.seed_base,
                "jitter_m": args.jitter,
                "closure_fractions": list(CLOSURE_FRACTIONS),
                "real_observed_fraction": REAL_OBSERVED_FRACTION,
                "policy": "scripted (천장)",
                "render": False,
                "prereg": "docs/PREREG_gripper_closure_0912.md",
            },
            result=payload,
        )

    return 0 if instrument_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
