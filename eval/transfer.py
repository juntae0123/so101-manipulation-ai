"""How fast does a sim-trained policy fall over when the domain moves?
도메인이 움직이면 시뮬 학습 정책은 얼마나 빨리 무너지는가?

This is the instrument for D-AI-18: whether simulation training is worth
anything as a head start for the real UMI demonstrations. The real question
needs the real arm and the arm is four weeks away, so this measures the weaker
question that sits inside it — the same policy, the same seeds, the same object
placements, evaluated under a nominal domain (A) and a perturbed one (B).
D-AI-18 의 계측기다 — 시뮬 학습이 실물 UMI 시연의 출발점으로 값어치가 있는가.
진짜 질문은 실물이 있어야 하고 실물은 4주 뒤다. 그래서 그 안에 든 약한 질문을
잰다. 같은 정책, 같은 시드, 같은 물체 배치를 공칭 도메인(A)과 흔든 도메인(B)에서
평가한다.

    붕괴율 = 1 - (B 성공률 / A 성공률)

⚠️ 이 수치는 실제 sim2real 갭의 **하한**이다. 마찰·센서 노이즈·조명·캘리브레이션
   오차를 시뮬은 재현하지 못한다. 결론은 한 방향으로만 유효하다 — 여기서 무너지면
   실물에서도 무너진다. 여기서 버틴다고 실물을 보장하지 않는다.

⚠️ 붕괴율은 나눗셈이다. 조건 A 성공률이 낮으면 분모가 작아 값이 요동친다.
   그래서 A 성공률 하한을 게이트로 먼저 두고, 못 넘으면 붕괴율을 보고하지 않는다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval.rollout import evaluate
from eval.stats import wilson_ci
from policy.base import Policy
from policy.baselines import ScriptedPickPolicy
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config
from sim.mujoco.domain import PRESETS, DomainRandomizer, DomainSpec
from sim.mujoco.env import MujocoPickEnv
from tracking.exp_log import file_digest, log_run

# Fixed before any transfer number existed. Changing a threshold after seeing
# the result is not a judgement, it is a rationalisation.
# 어떤 전이 수치도 나오기 전에 확정했다. 결과를 보고 문턱을 옮기면 판정이 아니라
# 사후 합리화다.
MIN_A_RATE = 0.20
ADOPT_BELOW = 0.30
REJECT_ATOR_ABOVE = 0.70

GATES: dict[str, str] = {
    "validity": (
        f"조건 A 성공률 >= {MIN_A_RATE:.0%}. 못 넘으면 붕괴율은 무효다 — "
        "분모가 0 에 가까워 나눗셈이 의미를 잃는다."
    ),
    "adopt": (
        f"붕괴율 <= {ADOPT_BELOW:.2f} → S3(시뮬 사전학습 → UMI 데이터 파인튜닝) "
        "시도 가치 있음"
    ),
    "retry": (
        f"{ADOPT_BELOW:.2f} < 붕괴율 < {REJECT_ATOR_ABOVE:.2f} → 도메인 랜덤화를 켜고 "
        "재수집·재학습한 뒤 재판정"
    ),
    "reject": (
        f"붕괴율 >= {REJECT_ATOR_ABOVE:.2f} → S3/S4 기각. 시뮬은 실행가능성 검사와 "
        "평가 harness 로만 쓰고 자원은 UMI 실데이터에 몰아준다"
    ),
}




@dataclass
class TransferRow:
    """One policy's result across both domain conditions.
    정책 하나의 두 도메인 조건에 걸친 결과."""

    policy: str
    rate_a: float
    rate_b: float
    uses_privileged_state: bool

    @property
    def collapse(self) -> float | None:
        """1 - B/A, or None when A is too low for the ratio to mean anything.
        1 - B/A. A 가 너무 낮아 비율이 의미를 잃으면 None."""
        if self.rate_a < MIN_A_RATE:
            return None
        return 1.0 - (self.rate_b / self.rate_a)

    def verdict(self) -> str:
        """Judgement against the gates fixed above.
        위에 고정된 게이트에 대한 판정.

        The adopt/reject gates are about a LEARNED policy's transfer. A scripted
        policy reads privileged state, so its collapse says something about the
        physics of the scene and nothing about whether sim pretraining is worth
        doing. Printing "S3 시도 가치 있음" beside it invites exactly the wrong
        reading, so privileged policies get no adoption verdict at all.
        채택/기각 게이트는 **학습 정책**의 전이에 관한 것이다. 스크립트 정책은
        특권 정보를 읽으므로 그 붕괴는 씬의 물리에 대해 말할 뿐, 시뮬 사전학습이
        값어치가 있는지에 대해서는 아무 말도 하지 않는다. 옆에 "S3 시도 가치
        있음"을 찍으면 정확히 틀린 독해를 부른다. 그래서 특권 정책에는 채택
        판정을 아예 붙이지 않는다.
        """
        c = self.collapse
        if self.uses_privileged_state:
            if c is None:
                return "참고용 — 특권 정보 정책, 채택 판정 대상 아님"
            return f"참고용 (물리 외란 민감도 {c:+.3f}) — 채택 판정 대상 아님"
        if c is None:
            return f"무효 — 조건 A 성공률 {self.rate_a:.0%} < {MIN_A_RATE:.0%}"
        if c <= ADOPT_BELOW:
            return "S3 시도 가치 있음"
        if c >= REJECT_ATOR_ABOVE:
            return "S3/S4 기각"
        return "랜덤화 켜고 재판정"


def _make_env(cfg: dict[str, Any], spec: DomainSpec, *, render: bool, jitter: float) -> MujocoPickEnv:
    """Build an env under one domain condition.
    도메인 조건 하나로 환경을 만든다."""
    return MujocoPickEnv(
        cfg, render=render, object_jitter_m=jitter, domain=DomainRandomizer(spec)
    )


def _policies(
    env: MujocoPickEnv, policy_ckpt: Path | None, *, with_scripted: bool = True
) -> list[Policy]:
    """Scripted always, plus the learned checkpoint when one is given.
    스크립트는 항상, 학습 체크포인트는 주어졌을 때.

    Scripted is here as the physics-only reference: it reads privileged state
    and never looks at an image, so whatever it loses under condition B is lost
    to friction and mass alone. A vision policy loses that plus perception.
    스크립트는 물리 전용 기준선으로 들어간다. 특권 정보를 읽고 이미지를 보지
    않으므로 조건 B 에서 잃는 것은 전부 마찰과 질량 탓이다. 시각 정책은 거기에
    인식까지 더해 잃는다.
    """
    # Two runs have now shown scripted scoring identically under A and B
    # (70.0% both times): its failures are IK-unreachable placements, which the
    # physics perturbation cannot touch. It is a dead probe here, and it costs
    # half the render time, so it can be dropped once that is established.
    # 두 번의 실행에서 scripted 가 A/B 동일 점수(둘 다 70.0%)를 냈다. 그 실패는
    # IK 도달 불가 배치이고 물리 외란이 건드릴 수 없는 종류다. 여기서는 죽은
    # 프로브이고 렌더 시간의 절반을 먹으므로, 확인된 이상 뺄 수 있다.
    out: list[Policy] = [ScriptedPickPolicy(env)] if with_scripted else []
    if policy_ckpt is not None:
        from policy.act import load_policy

        bc = load_policy(policy_ckpt)
        if bc.meta.get("trained_on") == "random_tensors":
            raise SystemExit(
                "이 체크포인트는 랜덤 텐서로 학습된 것이다. 전이 붕괴율을 재도 "
                "의미가 없다. 실데이터로 학습한 뒤 다시 실행하라."
            )
        print(f"학습 정책 로드: {bc.describe()}")
        out.append(bc)
    return out


def measure(
    cfg: dict[str, Any],
    seeds: list[int],
    *,
    policy_ckpt: Path | None,
    render: bool,
    jitter: float,
    with_scripted: bool = True,
) -> tuple[list[TransferRow], dict[str, Any], dict[str, Any]]:
    """Score every policy under condition A, then the same ones under B.
    정책 전부를 조건 A 에서 채점하고, 같은 것들을 조건 B 에서 채점한다."""
    rates: dict[str, dict[str, float]] = {}
    privileged: dict[str, bool] = {}
    domain_info: dict[str, Any] = {}
    # Per-trial rows are kept because the aggregate cannot answer "where does it
    # succeed". Re-running a 100-episode rendered sweep to recover a field we
    # already had in memory is the expensive kind of mistake.
    # 시행별 기록을 남긴다. 집계값은 "어디서 성공하는가"에 답하지 못한다. 이미
    # 메모리에 있던 값을 되찾겠다고 100편짜리 렌더 스윕을 다시 도는 것이 비싼
    # 종류의 실수다.
    detail: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for key in ("A", "B"):
        spec = PRESETS[key]
        with _make_env(cfg, spec, render=render, jitter=jitter) as env:
            randomiser = DomainRandomizer(spec)
            randomiser.bind(env.model)
            domain_info[key] = randomiser.describe()
            for policy in _policies(env, policy_ckpt, with_scripted=with_scripted):
                rate, results = evaluate(env, policy, seeds)
                rates.setdefault(policy.name, {})[key] = rate
                privileged[policy.name] = policy.uses_privileged_state
                detail.setdefault(policy.name, {})[key] = [r.__dict__ for r in results]
                print(f"  [{key}] {policy.name:10s} {rate * 100:5.1f}%")

    rows = [
        TransferRow(
            policy=name,
            rate_a=v["A"],
            rate_b=v["B"],
            uses_privileged_state=privileged[name],
        )
        for name, v in rates.items()
    ]
    return rows, domain_info, detail


def format_table(rows: list[TransferRow], n_episodes: int) -> str:
    """The table that goes into the MEASURE document verbatim.
    MEASURE 문서에 그대로 들어갈 표."""
    head = f"{'정책':12s} {'조건 A (95% CI)':>22s} {'조건 B (95% CI)':>22s} {'붕괴율':>9s}  판정"
    lines = [head, "-" * 100]
    for r in rows:
        c = r.collapse
        c_txt = "  —" if c is None else f"{c:+.3f}"
        mark = " *" if r.uses_privileged_state else "  "
        a_lo, a_hi = wilson_ci(round(r.rate_a * n_episodes), n_episodes)
        b_lo, b_hi = wilson_ci(round(r.rate_b * n_episodes), n_episodes)
        lines.append(
            f"{r.policy:12s}{mark} "
            f"{r.rate_a * 100:5.1f}% [{a_lo * 100:4.1f}~{a_hi * 100:5.1f}] "
            f"{r.rate_b * 100:5.1f}% [{b_lo * 100:4.1f}~{b_hi * 100:5.1f}] "
            f"{c_txt:>9s}  {r.verdict()}"
        )
    lines.append("* 특권 정보 사용 — 이미지를 보지 않으므로 물리 외란만 반영된다")
    lines.append(
        f"구간은 95% Wilson. n={n_episodes} 에서는 넓다 — 겹치는 두 값의 차이는 "
        "발견이 아니다."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--policy-ckpt", type=Path, default=None)
    parser.add_argument(
        "--render",
        action="store_true",
        help="관측 렌더링. 시각 정책을 채점하려면 반드시 켜야 한다",
    )
    parser.add_argument(
        "--skip-scripted",
        action="store_true",
        help="scripted 기준선을 빼고 학습 정책만 잰다. 렌더 시간이 절반이 된다",
    )
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    if args.skip_scripted and args.policy_ckpt is None:
        raise SystemExit("--skip-scripted 는 --policy-ckpt 와 함께 써야 한다. 잴 정책이 없다.")

    # Without rendering the observation images are zeros, so every appearance
    # perturbation has exactly zero effect and the number would look reassuring
    # for the wrong reason. Refuse rather than produce that.
    # 렌더링 없이는 관측 이미지가 0 이라 외관 외란이 정확히 무효가 되고, 수치는
    # 틀린 이유로 안심스럽게 보인다. 그런 값을 내느니 거부한다.
    if args.policy_ckpt is not None and not args.render:
        raise SystemExit(
            "시각 정책을 --render 없이 채점하면 이미지가 전부 0 이라 외관 외란이 "
            "무효가 된다. --render 를 붙여라."
        )

    cfg = load_config()
    seeds = [args.seed_base + i for i in range(args.episodes)]

    print("게이트 기준 (결과 확인 전 확정):")
    for key, text in GATES.items():
        print(f"  [{key}] {text}")
    print()

    rows, domain_info, detail = measure(
        cfg,
        seeds,
        policy_ckpt=args.policy_ckpt,
        render=args.render,
        jitter=args.jitter,
        with_scripted=not args.skip_scripted,
    )

    print()
    print(format_table(rows, args.episodes))
    print()
    print(
        f"조건: 물체 xy ±{args.jitter * 1000:.0f}mm, 시드 {seeds[0]}~{seeds[-1]}, "
        f"A/B 동일 시드·동일 물체 배치, 렌더 {'ON' if args.render else 'OFF'}"
    )
    print(f"조건 B 외란: {domain_info['B']['spec']}")
    if "warning" in domain_info["B"]:
        print(f"⚠️ {domain_info['B']['warning']}")

    by_name = {r.policy: r for r in rows}
    if "scripted" in by_name and "bc" in by_name:
        cs, cb = by_name["scripted"].collapse, by_name["bc"].collapse
        if cs is not None and cb is not None:
            print(
                f"\n인식이 치르는 비용 (bc 붕괴율 - scripted 붕괴율): {cb - cs:+.3f}  "
                "🟡 두 정책이 다른 알고리즘이라 깨끗한 분해가 아니다. 방향만 읽어라."
            )

    print(
        "\n⚠️ 이 붕괴율은 실제 sim2real 갭의 하한이다. 여기서 무너지면 실물에서도 "
        "무너지지만, 버틴다고 실물을 보장하지 않는다."
    )

    if args.log:
        rec = log_run(
            experiment="domain_transfer",
            author=args.author,
            issue="S15P21A103-65",
            conditions={
                "episodes": args.episodes,
                "seeds": [seeds[0], seeds[-1]],
                "jitter_m": args.jitter,
                "render": args.render,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "policy_ckpt": str(args.policy_ckpt) if args.policy_ckpt else None,
                "with_scripted": not args.skip_scripted,
                "domain": domain_info,
                "gates": GATES,
                "metric": "붕괴율 = 1 - (조건B 성공률 / 조건A 성공률)",
                "limitation": "시뮬 내부 A→B 는 실제 sim2real 갭의 하한이다",
            },
            result={
                "detail": detail,
                "rows": [
                    {
                        "policy": r.policy,
                        "rate_a": r.rate_a,
                        "rate_b": r.rate_b,
                        "collapse": r.collapse,
                        "verdict": r.verdict(),
                        "uses_privileged_state": r.uses_privileged_state,
                    }
                    for r in rows
                ]
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
