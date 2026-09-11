"""Train the same configuration several times and report the spread.
같은 설정으로 여러 번 학습하고 그 폭을 보고한다.

Why a single run cannot stand for a checkpoint.
왜 한 번의 실행이 체크포인트를 대표하지 못하는가.

On 2026-09-01 two trainings on identical data with identical settings scored 0%
and 25% on the same seed block. Reporting either number alone would have been a
claim the measurement does not support -- one would have said the policy fails,
the other that it passes the gate. The difference was training nondeterminism,
and nothing in the loss curve hinted at it.
2026-09-01, 동일한 데이터·동일한 설정의 학습 두 번이 같은 시드 블록에서 0% 와 25% 를
냈다. 둘 중 하나만 보고했다면 계측이 뒷받침하지 않는 주장이 됐을 것이다 — 하나는
정책이 실패한다고, 다른 하나는 게이트를 통과한다고 말했을 테니까. 차이는 학습의
비결정성이었고, 손실 곡선에는 아무 힌트도 없었다.

So the deployable-checkpoint gate requires `runs >= 3` and this tool produces it:
N trainings with different seeds, each scored under the same conditions, reported
as mean and range rather than a single number.
그래서 배포 가능 체크포인트 게이트는 `runs >= 3` 을 요구하고, 이 도구가 그것을
만든다. 서로 다른 시드로 N 회 학습하고 각각을 같은 조건에서 채점한 뒤, 단일 수치가
아니라 평균과 범위로 보고한다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contract.skills import GATE_MIN_RUNS, ROLLOUT_GATE
from eval.stats import wilson_ci
from paths import AI_ROOT, DEFAULT_EXP_LOG
from policy.bc import DEFAULT_TRAIN_CONFIG
from sim.mujoco.build_scene import DEFAULT_CONFIG
from tracking.exp_log import code_digest, file_digest, log_run
from tracking.findings import brief as findings_brief
from tracking.findings import write as findings_write
from tracking.monitor import print_drift


@dataclass
class RunResult:
    """One training and its rollout score.
    학습 한 번과 그 롤아웃 점수."""

    seed: int
    ckpt: Path
    rate: float
    n_episodes: int
    action_space: str
    val_loss: float


def _run(cmd: list[str]) -> None:
    """Run a tool as a subprocess so it logs to EXP_LOG exactly as it normally does.
    도구를 하위 프로세스로 돌린다. 평소와 똑같이 EXP_LOG 에 기록되게 하려는 것이다."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    proc = subprocess.run(cmd, cwd=AI_ROOT)
    if proc.returncode not in (0, 1):
        # 1 은 게이트 실패를 뜻한다. 그건 결과이지 오류가 아니다.
        raise SystemExit(f"명령이 종료코드 {proc.returncode} 로 실패했다: {cmd[0]}")


def _latest_record(experiment: str, match: dict[str, Any]) -> dict[str, Any] | None:
    """Newest EXP_LOG record whose conditions contain every key/value in `match`.
    conditions 가 `match` 의 모든 키·값을 담은 가장 최근 EXP_LOG 기록."""
    found: dict[str, Any] | None = None
    for line in DEFAULT_EXP_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("experiment") != experiment:
            continue
        cond = rec.get("conditions") or {}
        if all(cond.get(k) == v for k, v in match.items()):
            found = rec
    return found


# Captured when the module loads, i.e. when the run starts.
# `log_run` computes `code_sha` and `git_rev` at **log time**, which for a job
# that runs for hours is a different tree than the one that produced the number.
# Measured 2026-09-07 🟢: the v6 run logged git_rev b7ffabe -- a commit that did
# not exist when the job launched -- and its `train_config_sha` came out None
# because the loaded module predated that field. A record that names the wrong
# code is worse than one that names none.
# 모듈이 로드될 때, 즉 **실행이 시작될 때** 잡는다. `log_run` 은 `code_sha` 와
# `git_rev` 를 **로그 시점에** 계산하는데, 몇 시간 도는 잡에서는 그 트리가 수치를
# 만든 트리와 다르다. 2026-09-07 실측 🟢: v6 실행이 git_rev b7ffabe 로 기록됐는데
# 그 커밋은 잡이 시작될 때 존재하지 않았고, `train_config_sha` 가 None 으로 나온
# 것도 로드된 모듈이 그 필드보다 먼저였기 때문이다. 틀린 코드를 가리키는 기록은
# 아무것도 가리키지 않는 기록보다 나쁘다.
CODE_SHA_AT_LAUNCH = code_digest()


def _worktree_dirty() -> bool:
    """Are there uncommitted tracked changes right now?
    지금 커밋되지 않은 추적 변경이 있는가?"""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                             cwd=AI_ROOT, capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def run_conditions(args: Any, train_seeds: list[int] | None = None) -> dict[str, Any]:
    """The one place a repeat_runs condition set is built.
    repeat_runs 조건 묶음을 만드는 **유일한** 자리.

    Both the pre-run drift check and the EXP_LOG record read from here. They used
    to be two separate literals, and on 2026-09-07 the drift check reported "no
    change" for a run that had changed `image_noise_gray` and `tag` -- because
    those keys had been added to one literal and not the other. A drift detector
    that can itself drift is worse than none: it reports safety it has not checked.
    실행 전 표류 검사와 EXP_LOG 기록이 **둘 다** 여기서 읽는다. 예전에는 두 개의
    별도 리터럴이었고, 2026-09-07 에 `image_noise_gray` 와 `tag` 가 바뀐 실행을
    표류 검사가 "동일하다"로 보고했다 -- 한쪽 리터럴에만 키를 넣었기 때문이다.
    스스로 표류하는 표류 감지기는 없느니만 못하다. 검사하지 않은 안전을 보고한다."""
    cond: dict[str, Any] = {
        "runs": args.runs,
        "epochs": args.epochs,
        "episodes": args.episodes,
        "eval_seed_base": args.eval_seed_base,
        "jitter_m": args.jitter,
        "tag": args.tag,
        "image_noise_gray": args.image_noise,
        # 씬 설정과 **학습 설정을 따로** 해싱한다. 예전에는 씬만 해싱해서
        # bc.yaml 변경(epochs·lr·행동공간)이 표류 감지 밖에 있었다.
        "config_sha": file_digest(DEFAULT_CONFIG),
        "code_sha_at_launch": CODE_SHA_AT_LAUNCH,
        # 추론 장치. v2~v6 은 전부 cpu 였고 기록에 없었다 (eval/rollout.py 주석 참조).
        "policy_device": getattr(args, "policy_device", "cpu"),
        # 행동공간 덮어쓰기. train_config_sha 는 파일 해시라 이걸 반영하지 못한다.
        "action_space_override": getattr(args, "action_space", None),
        "cameras_override": getattr(args, "cameras", None),
        "train_config_sha": file_digest(DEFAULT_TRAIN_CONFIG),
        "gate": {"rollout": ROLLOUT_GATE, "min_runs": GATE_MIN_RUNS},
    }
    if train_seeds is not None:
        cond["data"] = str(args.data)
        cond["train_seeds"] = train_seeds
    return cond


def repeat(
    data: Path,
    *,
    runs: int,
    episodes: int,
    seed_base: int,
    eval_seed_base: int,
    device: str,
    jitter: float,
    epochs: int | None = None,
    tag: str = "",
    image_noise: float | None = None,
    policy_device: str = "cpu",
    action_space: str | None = None,
    cameras: str | None = None,
) -> list[RunResult]:
    """Train `runs` times, score each, and collect the results.
    `runs` 회 학습하고 각각 채점해 결과를 모은다."""
    out: list[RunResult] = []
    for i in range(runs):
        seed = seed_base + i
        # 체크포인트 이름이 데이터셋 이름에서만 파생되면, 같은 데이터로 다른 설정을
        # 돌릴 때 **조용히 덮어쓴다**. 2026-09-07 에 완주한 실험을 그렇게 잃었다.
        # `tag` 는 그 충돌을 막는 자리다.
        name = f"{data.name}{('_' + tag) if tag else ''}_seed{seed}.pt"
        ckpt = AI_ROOT / "checkpoints" / "bc" / name
        if ckpt.exists():
            print(f"⚠️ 덮어쓴다: {ckpt.name} (이미 있다). 보존하려면 --tag 를 바꿔라")

        train_cmd = [sys.executable, "tools/train_bc.py", "--data", str(data),
                     "--seed", str(seed), "--out", str(ckpt), "--device", device, "--log"]
        if epochs is not None:
            train_cmd += ["--epochs", str(epochs)]
        if image_noise is not None:
            train_cmd += ["--image-noise", str(image_noise)]
        if action_space is not None:
            train_cmd += ["--action-space", action_space]
        # 카메라 부분집합. 실물 배포 구성(손목 1대)을 재수집 없이 학습하기 위한 것이다.
        # L76 참조. 평가 환경은 여전히 2대를 렌더하고 정책이 필요한 것만 읽는다.
        if cameras is not None:
            train_cmd += ["--cameras", cameras]
        _run(train_cmd)
        # `--policy-device` 를 자식에게 넘긴다. 안 넘기면 repeat_runs 의 conditions 에는
        # 기록되는데 실제 평가는 eval_rollout 기본값으로 돌아 **기록과 실행이 갈린다.**
        # 오늘 같은 유형의 구멍(기록이 그 코드를 안 가리킴)을 이미 한 번 냈다.
        _run([sys.executable, "tools/eval_rollout.py", "--episodes", str(episodes),
              "--seed-base", str(eval_seed_base), "--jitter", str(jitter), "--render",
              "--policy-device", policy_device,
              "--policy-ckpt", str(ckpt), "--log"])

        roll = _latest_record("rollout_baselines", {"policy_ckpt": str(ckpt)})
        train = _latest_record("train_bc", {"trained_on": str(data), "seed": seed})
        if roll is None:
            raise SystemExit(
                f"EXP_LOG 에서 {ckpt.name} 의 롤아웃 기록을 찾지 못했다. "
                "eval_rollout 이 --log 로 돌았는지 확인하라."
            )
        rate = float(roll["result"]["success_rates"]["bc"])
        out.append(RunResult(
            seed=seed,
            ckpt=ckpt,
            rate=rate,
            n_episodes=int(roll["conditions"]["episodes"]),
            action_space=str(roll["conditions"].get("policy_action_space", "?")),
            val_loss=float((train or {}).get("result", {}).get("best_val_loss", float("nan"))),
        ))
    return out


def summarise(results: list[RunResult]) -> dict[str, Any]:
    """Mean, range, and a pooled interval — with the pooling caveat attached.
    평균·범위와 합산 구간. 합산의 한계를 함께 붙인다."""
    rates = [r.rate for r in results]
    n_each = results[0].n_episodes
    total_n = n_each * len(results)
    total_ok = int(round(sum(rates) * n_each))
    lo, hi = wilson_ci(total_ok, total_n)
    return {
        "runs": len(results),
        "episodes_each": n_each,
        "mean": float(np.mean(rates)),
        "min": float(np.min(rates)),
        "max": float(np.max(rates)),
        "spread": float(np.max(rates) - np.min(rates)),
        "pooled_successes": total_ok,
        "pooled_n": total_n,
        "ci95": [lo, hi],
        "ci95_caveat": (
            "구간은 실행 전체를 합산해 계산했다. 서로 다른 학습은 엄밀히는 같은 "
            "정책의 표본이 아니므로, 이 구간은 '이 설정이 내놓는 정책'의 구간으로 "
            "읽어야 한다. 개별 체크포인트의 구간이 아니다."
        ),
    }


def format_report(results: list[RunResult], summary: dict[str, Any]) -> str:
    lines = [
        f"{'시드':>6s} {'행동공간':>16s} {'val_loss':>10s} {'롤아웃':>9s}  체크포인트",
        "-" * 78,
    ]
    for r in results:
        lines.append(
            f"{r.seed:6d} {r.action_space:>16s} {r.val_loss:10.5f} "
            f"{r.rate * 100:8.1f}%  {r.ckpt.name}"
        )
    lines.append("-" * 78)
    lines.append(
        f"{'평균':>6s} {'':>16s} {'':>10s} {summary['mean'] * 100:8.1f}%   "
        f"범위 {summary['min'] * 100:.1f}~{summary['max'] * 100:.1f}% "
        f"(폭 {summary['spread'] * 100:.1f}%p)"
    )
    lines.append(
        f"{'합산':>6s} {'':>16s} {'':>10s} "
        f"{summary['pooled_successes']}/{summary['pooled_n']}   "
        f"95% 구간 {summary['ci95'][0] * 100:.1f}~{summary['ci95'][1] * 100:.1f}%"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=GATE_MIN_RUNS)
    parser.add_argument("--epochs", type=int, default=None,
                        help="train_bc 에 그대로 넘긴다. 없으면 설정값(configs/train/bc.yaml)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=0, help="학습 시드 시작값")
    parser.add_argument("--eval-seed-base", type=int, default=3000, help="평가 시드 블록")
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--tag", type=str, default="",
                        help="체크포인트 이름에 붙일 꼬리표. 같은 데이터로 다른 설정을 "
                             "돌릴 때 덮어쓰기를 막는다")
    parser.add_argument("--image-noise", type=float, default=None,
                        metavar="GRAY", help="학습 이미지 잡음 σ, 단위 계조. train_bc 로 전달")
    parser.add_argument("--device", type=str, default="cuda",
                        help="학습 장치")
    parser.add_argument("--policy-device", type=str, default="cpu",
                        help="평가 시 정책 추론 장치. 기본 cpu 로 v2~v6 비교선 유지")
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument(
        "--action-space", type=str, default=None,
        help="model.action_space 를 덮어쓴다 (자식 train_bc 로 전달)",
    )
    parser.add_argument(
        "--cameras", type=str, default=None,
        help="쉼표로 구분한 카메라 부분집합 (자식 train_bc 로 전달). 예: cam_wrist",
    )
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    if args.runs < GATE_MIN_RUNS:
        print(
            f"⚠️ runs={args.runs} 는 배포 게이트의 최소치 {GATE_MIN_RUNS} 보다 작다. "
            "이 결과로는 배포 판정을 할 수 없다."
        )

    # 조건 표류를 실행 **전에** 찍는다. 결과를 다 뽑은 뒤에 알면 늦다.
    print_drift(run_conditions(args))

    # 지금까지 기록된 repeat_runs 5건이 **전부** dirty=True 였다 🟢 2026-09-07.
    # 그래서 v2→v3→v5 의 차이를 어느 변경에 귀속시킬 수 없다. 커밋 해시가
    # 가리키는 트리에서 돈 것이 아니기 때문이다.
    if _worktree_dirty():
        print("⚠️ 커밋되지 않은 변경이 있는 트리에서 돈다. 이 수치는 git 커밋으로 "
              "되짚어갈 수 없고, code_sha 로만 특정된다. 조건 비교가 목적이면 "
              "먼저 커밋하라")

    # 지난 실험 성적을 **시작 전에** 찍는다. 끝난 뒤에 대조하면 이미 조건을 정한 뒤다.
    print("\n" + findings_brief() + "\n")

    print(f"학습 {args.runs}회 × 롤아웃 {args.episodes}편 · 데이터 {args.data}")
    print(f"평가 시드 블록 {args.eval_seed_base}~{args.eval_seed_base + args.episodes - 1} "
          "(모든 실행이 동일)\n")

    results = repeat(
        args.data,
        runs=args.runs,
        epochs=args.epochs,
        episodes=args.episodes,
        seed_base=args.seed_base,
        eval_seed_base=args.eval_seed_base,
        device=args.device,
        jitter=args.jitter,
        tag=args.tag,
        image_noise=args.image_noise,
        policy_device=args.policy_device,
        action_space=args.action_space,
        cameras=args.cameras,
    )
    summary = summarise(results)

    print("\n" + "=" * 78)
    print(format_report(results, summary))
    print()

    spaces = {r.action_space for r in results}
    if len(spaces) > 1:
        print(f"⚠️ 행동 공간이 섞였다: {spaces}. 같은 조건의 반복이 아니다.")

    mean_ok = summary["mean"] > ROLLOUT_GATE
    ci_ok = summary["ci95"][0] > ROLLOUT_GATE
    runs_ok = summary["runs"] >= GATE_MIN_RUNS
    passed = mean_ok and ci_ok and runs_ok

    print("배포 게이트 판정 (contract/skills.py 의 상수):")
    print(f"  평균 {summary['mean']:.3f} > {ROLLOUT_GATE:.2f} → {'통과' if mean_ok else '실패'}")
    print(f"  95% 구간 하한 {summary['ci95'][0]:.3f} > {ROLLOUT_GATE:.2f} → "
          f"{'통과' if ci_ok else '실패'}")
    print(f"  실행 {summary['runs']} >= {GATE_MIN_RUNS} → {'통과' if runs_ok else '실패'}")
    print(f"\n→ {'배포 가능' if passed else '배포 불가'}")
    print(f"\n⚠️ {summary['ci95_caveat']}")

    if args.log:
        rec = log_run(
            experiment="repeat_runs",
            author=args.author,
            issue="S15P21A103-34",
            conditions=run_conditions(args, [r.seed for r in results]),
            result={
                **summary,
                "passed": passed,
                "per_run": [
                    {"seed": r.seed, "rate": r.rate, "val_loss": r.val_loss,
                     "action_space": r.action_space, "ckpt": str(r.ckpt)}
                    for r in results
                ],
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
        # 기록 직후 원장을 다시 생성한다. 손으로 갱신하면 갱신 안 된 상태가 기본값이 된다.
        print(f"FINDINGS.md 갱신: {findings_write()}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
