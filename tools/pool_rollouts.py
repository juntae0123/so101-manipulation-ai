#!/usr/bin/env python3
"""Pool per-seed rollout rows across every run log and report Wilson intervals.
모든 실행 로그의 시드별 롤아웃 행을 조건별로 합산해 Wilson 구간으로 보고한다.

로그의 요약표 행을 읽는다:
     0 joint_delta_gripper_binary    0.12307     10.0%  sim_pick_cmd_cmd_noise96_seed0.pt

조건 이름은 체크포인트 이름에서 뽑는다 (데이터셋명 + 태그). 태그가 조건이므로
어느 로그에 무엇이 들어 있는지 몰라도 된다.
"""
from __future__ import annotations

import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROW = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+([\d.]+|nan)\s+([\d.]+)%\s+(\S+\.pt)\s*$"
)
# train_bc 가 매 실행 머리말에 찍는 기준선. **조건마다 다르다** — chunk=K 는 손실이
# K 스텝에 걸쳐 계산돼 스케일이 다르고, 데이터셋마다 라벨 분포도 다르다.
# 2026-09-15 에 고정 상수 2.55132 로 비교하다 chunk=8 조건 14건을 전부 오탐으로
# 표시했다. 로그가 실제 값을 들고 있으니 그것을 읽는다.
TRIVIAL = re.compile(r"자명한 예측기.*?=\s*([\d.]+)")
N_PER_RUN = 100


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval. 작은 k 에서 정규근사보다 정직하다."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


def condition(ckpt: str) -> str:
    """sim_pick_cmd_cmd_noise96_seed0.pt -> sim_pick_cmd_cmd_noise96"""
    return re.sub(r"_seed\d+\.pt$", "", ckpt)


def main(argv: list[str]) -> int:
    logdir = Path(argv[1]) if len(argv) > 1 else Path("out/logs")
    rows: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    trivial: dict[str, float] = {}
    for log in sorted(logdir.glob("*.log")):
        text = log.read_text(errors="replace")
        # 이 로그가 스스로 찍은 기준선. 여러 시드가 같은 값을 찍으므로 첫 값을 쓴다.
        tm = TRIVIAL.search(text)
        base = float(tm.group(1)) if tm else float("nan")
        for line in text.splitlines():
            m = ROW.match(line)
            if not m:
                continue
            seed, _space, val, pct, ckpt = m.groups()
            v = float("nan") if val == "nan" else float(val)
            # 같은 시드가 여러 로그에 있으면 마지막 것을 쓴다 (재실행분)
            rows[condition(ckpt)][int(seed)] = (v, float(pct))
            trivial[condition(ckpt)] = base

    if not rows:
        print(f"!! 요약표 행을 못 찾았다: {logdir}/*.log")
        return 1

    out = []
    out.append(f"# 롤아웃 합산 — {logdir}")
    out.append("")
    out.append("조건당 시드별 n=100. 구간은 Wilson 95%.")
    out.append("")
    out.append(f"| 조건 | 실행수 | 성공/전체 | 성공률 | 95% 구간 | 시드별 | val 범위 |")
    out.append("|---|---:|---:|---:|---|---|---|")
    for cond in sorted(rows):
        per = rows[cond]
        seeds = sorted(per)
        succ = sum(round(per[s][1]) for s in seeds)
        n = len(seeds) * N_PER_RUN
        lo, hi = wilson(succ, n)
        pcts = " / ".join(f"{per[s][1]:.0f}" for s in seeds)
        vals = [per[s][0] for s in seeds if not math.isnan(per[s][0])]
        vr = f"{min(vals):.3f}~{max(vals):.3f}" if vals else "nan"
        out.append(
            f"| `{cond}` | {len(seeds)} | {succ}/{n} | {succ / n:.1%} | "
            f"[{lo:.1%}, {hi:.1%}] | {pcts} | {vr} |"
        )

    out.append("")
    out.append("## 자동 점검")
    out.append("")
    flagged = False
    for cond in sorted(rows):
        base = trivial.get(cond, float("nan"))
        for s, (v, p) in sorted(rows[cond].items()):
            if not math.isnan(v) and not math.isnan(base) and v > base * 0.9:
                out.append(
                    f"- 🔴 `{cond}` 시드 {s}: val {v:.3f} 가 이 조건의 자명한 예측기"
                    f"({base:.3f}) 수준이거나 그 이상이다. "
                    f"**학습 실패이지 조건 효과가 아니다**"
                )
                flagged = True
            if math.isnan(v):
                out.append(f"- 🔴 `{cond}` 시드 {s}: val nan")
                flagged = True
    if not flagged:
        out.append("- 자명한 예측기 수준 / nan 인 실행 없음")
    out.append("")
    out.append("조건별 자명한 예측기 손실 (로그에서 읽음):")
    for cond in sorted(rows):
        b = trivial.get(cond, float("nan"))
        out.append(f"  - `{cond}`: {b:.5f}" if not math.isnan(b) else f"  - `{cond}`: 못 읽음")

    out.append("")
    out.append("## 읽는 법")
    out.append("")
    out.append("- **구간이 겹치면 차이를 주장하지 않는다.** 겹치면 n 을 올린다")
    out.append("- 위에서 🔴 로 걸린 시드가 있으면 그 조건의 합산값은 오염돼 있다")
    out.append("- 전부 시뮬이다. 시뮬 성공률은 sim2real 갭의 하한이다")

    text = "\n".join(out)
    print(text)
    Path("out/ANALYSIS_latest.md").write_text(text + "\n", encoding="utf-8")
    print("\n저장: out/ANALYSIS_latest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
