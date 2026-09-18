"""Cross-check two independent gripper.csv producers frame by frame.
두 독립 구현의 gripper.csv 를 프레임 단위로 대조한다.

왜 있는가 / Why this exists.

같은 영상에서 gap 을 뽑는 구현이 둘 생겼다 (황도경 원본 · 김준태 재구현).
둘 중 하나를 정본으로 골라야 하는데, **단일 에피소드 fixture 로는 못 고른다** —
2026-09-12 에 단일 편 fixture 가 중앙오차 0.348mm 로 통과했지만 71편 전수에서는
1.26mm 였다. fixture 로 쓴 편이 보정상수를 적합한 바로 그 편이었기 때문이다.

그래서 이 도구는 **전수**로만 답한다. 출력 4가지:
  1) 검출 일치도 (둘 다 D / 한쪽만 D / 둘 다 X)
  2) gap 수치 차이 분포 + 편별 비율 범위  ← 전역 상수로 맞출 수 있는지 판정
  3) 미검출 프레임의 에피소드 내 위치 분포  ← 필터가 구조적인지 무차별인지 판정
  4) gap 절대 분포  ← "그리퍼가 실제로 닫히는가" 판정

(4) 가 2026-09-12 에 실제로 잡아낸 것이다: 5,759 프레임 중 gap<30mm 가 5개였다.
일치도만 봤으면 못 봤다. **두 계측기가 서로 일치한다고 둘 다 맞는 것은 아니다.**

사용:
    # [로컬]
    python AI/tools/compare_gripper_csv.py \
        --a ~/Downloads/gripper_csv_20260911_71ep \
        --b ~/Downloads/학습데이터 \
        --a-name gripper.csv --b-name gripper_reimpl.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import median

if hasattr(sys.stdout, "reconfigure"):          # Git Bash cp949 대비
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Series = dict[int, tuple[float | None, str]]

CLOSE_THRESHOLD_M = 0.035
"""폐쇄 전이 판정 임계. 개방 약 0.065m / 평탄 약 0.039m 사이를 잡으려 했으나
실측상 71편 중 68편이 이 값 아래로 내려가지 않는다 — 임계가 아니라 데이터 문제다."""

CLOSE_RUN = 5
"""임계 아래로 연속 몇 프레임이어야 폐쇄로 보는가. 단발 검출오차를 거른다."""


def load(path: Path) -> Series:
    """Read one gripper.csv. Values only count when status is D or M.
    gripper.csv 한 편을 읽는다. D·M 만 값으로 센다 (T·X 는 결측)."""
    out: Series = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            idx = int(row["frame_index"])
            status = (row.get("status") or "X").strip()[:1] or "X"
            raw = (row.get("gap_m") or "").strip()
            value = float(raw) if raw and status in ("D", "M") else None
            out[idx] = (value, status)
    return out


def close_index(series: Series, threshold: float = CLOSE_THRESHOLD_M) -> int | None:
    """First frame that goes below threshold and stays there for CLOSE_RUN frames.
    임계 아래로 내려가 CLOSE_RUN 프레임 유지하는 첫 프레임. 없으면 None."""
    keys = sorted(series)
    for n, i in enumerate(keys):
        value = series[i][0]
        if value is None or value >= threshold:
            continue
        window = [series[j][0] for j in keys[n:n + CLOSE_RUN]]
        if len(window) == CLOSE_RUN and all(
            v is not None and v < threshold for v in window
        ):
            return i
    return None


def quantile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    return sorted_values[min(len(sorted_values) - 1, int(p * len(sorted_values)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--a", type=Path, required=True, help="기준 산출물 루트 (rec_* 폴더들을 담은 디렉터리)")
    ap.add_argument("--b", type=Path, required=True, help="대조 산출물 루트")
    ap.add_argument("--a-name", default="gripper.csv")
    ap.add_argument("--b-name", default="gripper_reimpl.csv")
    ap.add_argument("--worst", type=int, default=15, help="편별 표에 찍을 상위 편 수")
    args = ap.parse_args()

    episodes = sorted(p.name for p in args.a.iterdir() if p.is_dir())
    if not episodes:
        print(f"에피소드가 없다: {args.a}", file=sys.stderr)
        return 2

    rows: list[tuple[str, int, int, int, int, float | None, float | None]] = []
    total = dict(frames=0, both=0, a_only=0, b_only=0, neither=0,
                 a_d=0, b_d=0, index_mismatch=0, missing=0, skipped=0)
    diffs_all: list[float] = []
    ratios_ep: list[float] = []
    gaps_a: list[float] = []
    ep_min_a: list[tuple[str, float]] = []
    pos_a: list[float] = []
    pos_b: list[float] = []
    close_agree = close_disagree = close_undecidable = 0
    close_bad: list[tuple[str, int, int]] = []

    for ep in episodes:
        pa, pb = args.a / ep / args.a_name, args.b / ep / args.b_name
        if not pa.exists():
            # 기준 파일이 없는 디렉터리는 에피소드가 아니다 (`_stale_v1` 등 작업 폴더).
            # B 쪽 누락으로 세면 "데이터가 빈다"로 오독된다.
            total["skipped"] += 1
            continue
        if not pb.exists():
            total["missing"] += 1
            continue
        A, B = load(pa), load(pb)
        if set(A) != set(B):
            total["index_mismatch"] += 1
        keys = sorted(set(A) & set(B))
        n = len(keys)
        diffs: list[float] = []
        ratios: list[float] = []
        n_both = n_a = n_b = n_none = 0

        for i in keys:
            ga, sa = A[i]
            gb, sb = B[i]
            da, db = ga is not None, gb is not None
            if da and db:
                n_both += 1
                diffs.append(abs(ga - gb) * 1000.0)
                if ga > 0:
                    ratios.append(gb / ga)
            elif da:
                n_a += 1
            elif db:
                n_b += 1
            else:
                n_none += 1

        total["frames"] += n
        total["both"] += n_both
        total["a_only"] += n_a
        total["b_only"] += n_b
        total["neither"] += n_none
        total["a_d"] += sum(1 for i in keys if A[i][0] is not None)
        total["b_d"] += sum(1 for i in keys if B[i][0] is not None)
        diffs_all += diffs
        if ratios:
            ratios_ep.append(median(ratios))

        values_a = [A[i][0] for i in keys if A[i][0] is not None]
        if values_a:
            gaps_a += [v * 1000.0 for v in values_a]
            ep_min_a.append((ep, min(values_a) * 1000.0))

        span = max(n - 1, 1)
        pos_a += [i / span for i in keys if A[i][0] is None]
        pos_b += [i / span for i in keys if B[i][0] is None]

        ca, cb = close_index(A), close_index(B)
        if ca is None or cb is None:
            close_undecidable += 1
        elif abs(ca - cb) <= 2:
            close_agree += 1
        else:
            close_disagree += 1
            close_bad.append((ep, ca, cb))

        rows.append((ep, n, n_both, n_a, n_b,
                     median(diffs) if diffs else None,
                     max(diffs) if diffs else None))

    frames = max(total["frames"], 1)
    bar = "=" * 78

    print(bar)
    print(f"편별 — 최대 차이 상위 {args.worst}")
    print(bar)
    print(f"{'episode':<24}{'frames':>7}{'both':>7}{'A만':>7}{'B만':>7}"
          f"{'중앙mm':>9}{'최대mm':>9}")
    for ep, n, nb, na, nbo, med, mx in sorted(
        rows, key=lambda r: -(r[6] or 0.0)
    )[: args.worst]:
        s_med = f"{med:.3f}" if med is not None else "-"
        s_max = f"{mx:.3f}" if mx is not None else "-"
        print(f"{ep:<24}{n:>7}{nb:>7}{na:>7}{nbo:>7}{s_med:>9}{s_max:>9}")

    print()
    print(bar)
    print("1) 검출 일치도")
    print(bar)
    n_ep = len(episodes) - total["skipped"]
    print(f"에피소드            : {n_ep}"
          f"  (B 쪽 누락 {total['missing']} · 기준파일 없어 제외 {total['skipped']})")
    print(f"공통 프레임         : {total['frames']}")
    print(f"A 검출              : {total['a_d']} ({total['a_d']/frames:.1%})")
    print(f"B 검출              : {total['b_d']} ({total['b_d']/frames:.1%})")
    print(f"둘 다 검출          : {total['both']} ({total['both']/frames:.1%})")
    print(f"A 만 (B 가 놓침)    : {total['a_only']}")
    print(f"B 만 (A 가 놓침)    : {total['b_only']}")
    print(f"둘 다 미검출        : {total['neither']}")
    print(f"프레임 인덱스 불일치 편: {total['index_mismatch']}")

    if diffs_all:
        ds = sorted(diffs_all)
        print()
        print(bar)
        print(f"2) gap 차이 (둘 다 검출, n={len(ds)})")
        print(bar)
        print(f"  중앙 {median(ds):.4f}  p90 {quantile(ds,.90):.4f}"
              f"  p99 {quantile(ds,.99):.4f}  최대 {ds[-1]:.4f} mm")
        over1 = sum(1 for d in ds if d > 1.0)
        print(f"  >1mm {over1} ({over1/len(ds):.2%})  >5mm {sum(1 for d in ds if d>5.0)}")
    if ratios_ep:
        lo, hi = min(ratios_ep), max(ratios_ep)
        print(f"  편별 비율(B/A) 중앙 {median(ratios_ep):.5f}  범위 {lo:.5f} ~ {hi:.5f}")
        if hi - lo > 0.05:
            print("  ⚠️ 편별 비율이 5%p 넘게 벌어진다 — 전역 보정상수로는 못 맞춘다.")
            print("     척도 문제가 아니라 프레임 선택 문제다.")

    print()
    print(bar)
    print("3) 미검출 프레임 위치 (에피소드 진행률 10분위)")
    print(bar)
    for label, xs in (("A", pos_a), ("B", pos_b)):
        bins = [0] * 10
        for x in xs:
            bins[min(9, int(x * 10))] += 1
        print(f"  {label}: {bins}  합계 {len(xs)}")
    print("  앞쪽에 몰리면 구조적(손 진입 등), 균일하면 무차별 폐기다.")

    if gaps_a:
        ga = sorted(gaps_a)
        print()
        print(bar)
        print(f"4) gap 절대 분포 (A 기준, n={len(ga)})")
        print(bar)
        print(f"  p01 {quantile(ga,.01):.1f}  p05 {quantile(ga,.05):.1f}"
              f"  중앙 {median(ga):.1f}  p95 {quantile(ga,.95):.1f}  최대 {ga[-1]:.1f} mm")
        for thr in (10, 20, 30, 35):
            c = sum(1 for g in ga if g < thr)
            print(f"  gap < {thr:>2}mm : {c} ({c/len(ga):.2%})")
        if ep_min_a:
            mins = [m for _, m in ep_min_a]
            print(f"  편별 최소 gap 중앙 {median(mins):.2f}mm"
                  f"  (최소 {min(mins):.2f} / 최대 {max(mins):.2f})")
        closed = sum(1 for g in ga if g < 30.0)
        if closed / len(ga) < 0.01:
            print("  🛑 gap<30mm 가 1% 미만이다 — 이 데이터셋에서 그리퍼가 닫히지 않는다.")
            print("     대상물 규격 15~25mm 와 맞지 않는다. 수집 프로토콜 또는")
            print("     폐쇄측 캘리브레이션 점을 판정해야 한다 (대상물 문 프레임 1장).")

    print()
    print(bar)
    print(f"5) 폐쇄 전이 프레임 (임계 {CLOSE_THRESHOLD_M*1000:.0f}mm · ±2프레임 허용)")
    print(bar)
    print(f"  일치 {close_agree} · 불일치 {close_disagree} · 판정불가 {close_undecidable}")
    for ep, ca, cb in close_bad:
        print(f"    {ep}  A {ca}  B {cb}")
    if close_undecidable > n_ep // 2:
        print("  ⚠️ 절반 이상이 판정불가다 — (4) 의 폐쇄 미도달과 같은 원인이다.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
