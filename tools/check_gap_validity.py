"""Judge whether a gripper gap channel is usable, not merely present.
gap 채널이 "있다"가 아니라 "쓸 만한가"를 판정한다.

왜 (2026-09-18)
---------------
행 수를 세는 검사로는 이 채널의 고장을 못 잡는다. 실증이 둘 있다.

- 2026-09-12: 도경 `gripper.csv` 71편은 **전부 데이터가 있었고 검출률 94.6%** 였다.
  그런데 gap 이 38~40mm 에서 평평해져 멈췄고, 그리퍼 채널을 학습에 쓸 수 없었다.
- 2026-09-18: 미검출 프레임도 `status=X` + `gap_m` 빈칸으로 **행은 그대로 나온다.**
  검출 0건 에피소드도 `rows == frames` 를 통과한다.

**검출 수와 유효성은 다른 양이다.** 이 도구는 후자만 본다. 구조 검사는 BE 몫이다.

게이트 — 데이터 보기 전에 확정했다 (2026-09-18)
------------------------------------------------
    G1 이봉성   두 봉우리 사이 골 깊이 / 작은 쪽 봉우리 높이 <= 0.35
                단봉이면 골이 없어 1 에 가깝다. 그리퍼가 안 움직였거나 변환이 상수인 경우다
                (초판의 Otsu 분리도 >= 2.0 은 자체검증이 기각했다 — 아래 주석)
    G2 폐쇄측   status=D 행만, 편별 최소 gap 의 중앙값
    G3 대조     |G2 - 물체 실폭| > 10mm  →  불합격. gap 채널을 학습에 넣지 않는다
                물체 실폭 미제공  →  **판정 불가**. 통과로 처리하지 않는다

⚠️ G3 의 10mm 는 hand-eye 상수 편향 허용폭에서 가져왔다
   (프로젝트 기록: ±10mm 에서 재생 3/4). 임의로 고른 수가 아니다.

정답을 아는 코퍼스
------------------
`~/Downloads/학습데이터` 0911 71편. 답을 이미 안다 (2026-09-12 전수 측정 🟢):
    편별 최소 gap 중앙  37.26 mm
    물체 실폭           41 mm (김현석 실측, 2026-09-18)
    따라서 G3 차이      3.74 mm  →  **통과해야 한다**
    봉우리 65mm / 39mm  →  G1 **통과해야 한다**
이 코퍼스에서 G1·G2·G3 가 위 값을 못 내면 계측기가 틀린 것이다.

Usage
-----
  python check_gap_validity.py --selftest
  python check_gap_validity.py --root <에피소드 루트> --object-width-mm 41
  python check_gap_validity.py --root <에피소드 루트>          # G3 판정 불가로 나온다
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

G1_MAX_VALLEY = 0.35      # 골 깊이 비. 낮을수록 확실한 이봉
G3_MAX_DIFF_MM = 10.0

# ⚠️ G1 초판은 Otsu 분리도 >= 2.0 이었다. **자체검증이 기각했다** —
#    Otsu 는 단봉도 무조건 둘로 쪼개므로 단봉 정규분포에서 분리도 2.63 이 나와
#    게이트를 통과했다(2026-09-18). 분리도는 진단값으로만 남기고,
#    판정은 두 봉우리 사이 **골 깊이**로 한다. 단봉에는 골이 없다.


# ── 순수 계산 — 자체검증 대상 ─────────────────────────────────────────────

def otsu_threshold(vals: list[float], bins: int = 64) -> float | None:
    """1-D Otsu split point. 1차원 오츠 임계. 나눌 수 없으면 None."""
    if len(vals) < 4:
        return None
    hist, edges = np.histogram(vals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    best_t, best_var = None, -1.0
    for i in range(1, bins):
        w0, w1 = hist[:i].sum(), hist[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = float((hist[:i] * centers[:i]).sum() / w0)
        m1 = float((hist[i:] * centers[i:]).sum() / w1)
        var = float(w0) * float(w1) * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, float(edges[i])
    return best_t


def separation(vals: list[float]) -> tuple[float, float, float, float]:
    """Cluster separation after an Otsu split.
    오츠 분할 후 분리도. 반환 (분리도, 저군집 평균, 고군집 평균, 합산표준편차)."""
    t = otsu_threshold(vals)
    if t is None:
        return (0.0, 0.0, 0.0, 0.0)
    a = [v for v in vals if v <= t]
    b = [v for v in vals if v > t]
    if len(a) < 2 or len(b) < 2:
        return (0.0, 0.0, 0.0, 0.0)
    sa, sb = st.pstdev(a), st.pstdev(b)
    pooled = ((sa ** 2 + sb ** 2) / 2) ** 0.5
    ma, mb = st.mean(a), st.mean(b)
    if pooled <= 0:
        return (float("inf"), ma, mb, 0.0)
    return (abs(mb - ma) / pooled, ma, mb, pooled)


def valley_ratio(vals: list[float], bins: int = 64) -> tuple[float, float, float]:
    """Depth of the trough between the two modes, relative to the smaller mode.
    두 봉우리 사이 골의 깊이를 작은 쪽 봉우리로 나눈 값.
    단봉이면 골이 없어 1 에 가깝고, 확실한 이봉이면 0 에 가깝다.
    반환 (골 비율, 저봉 위치, 고봉 위치). 판정 불가면 (1.0, 0, 0)."""
    if len(vals) < 8:
        return (1.0, 0.0, 0.0)
    t = otsu_threshold(vals)
    if t is None:
        return (1.0, 0.0, 0.0)
    hist, edges = np.histogram(vals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    i = int(np.searchsorted(edges, t)) - 1
    i = max(1, min(bins - 1, i))
    if hist[:i].sum() == 0 or hist[i:].sum() == 0:
        return (1.0, 0.0, 0.0)
    li = int(np.argmax(hist[:i]))
    ri = int(np.argmax(hist[i:])) + i
    if ri <= li:
        return (1.0, 0.0, 0.0)
    lo, hi = int(hist[li]), int(hist[ri])
    valley = int(hist[li:ri + 1].min())
    small = min(lo, hi)
    if small == 0:
        return (1.0, 0.0, 0.0)
    return (valley / small, float(centers[li]), float(centers[ri]))


def per_episode_min(ep_gaps: list[list[float]]) -> float | None:
    """Median of per-episode minimum gap. 편별 최소 gap 의 중앙값."""
    mins = [min(g) for g in ep_gaps if g]
    return st.median(mins) if mins else None


def longest_run_false(flags: list[bool]) -> int:
    """Longest run of False. False 가 연속된 최대 길이."""
    best = cur = 0
    for f in flags:
        cur = 0 if f else cur + 1
        best = max(best, cur)
    return best


# ── 자체검증 ─────────────────────────────────────────────────────────────

def selftest() -> int:
    bad = 0
    rng = np.random.default_rng(0)

    lo = list(rng.normal(39.0, 1.0, 400)); hi = list(rng.normal(65.0, 1.0, 400))
    v, a1, b1 = valley_ratio(lo + hi); sep, *_ = separation(lo + hi)
    ok = v <= G1_MAX_VALLEY
    print(f"[1] 이봉 합성(39/65) 골 → {v:.3f} (봉 {a1:.1f}/{b1:.1f}, 분리도 {sep:.2f})  "
          f"게이트 <= {G1_MAX_VALLEY}  ", end="")
    print("OK" if ok else "!! 실패 — 명백한 이봉을 못 잡는다"); bad += (not ok)

    v, *_ = valley_ratio(list(rng.normal(50.0, 1.0, 800))); sep, *_ = separation(list(rng.normal(50.0, 1.0, 800)))
    ok = v > G1_MAX_VALLEY
    print(f"[2] 단봉 합성(50) 골 → {v:.3f} (분리도 {sep:.2f})  게이트 초과여야  ", end="")
    print("OK" if ok else "!! 실패 — 단봉을 이봉이라 한다"); bad += (not ok)

    v, *_ = valley_ratio([50.0] * 500)
    ok = v > G1_MAX_VALLEY
    print(f"[3] 완전 상수 골 → {v:.3f}  게이트 초과여야  ", end="")
    print("OK" if ok else "!! 실패 — 상수를 이봉이라 한다. gap 이 안 움직인 편을 통과시킨다")
    bad += (not ok)

    # ⚠️ 초판 [3b] 는 39/43(4시그마)을 "겹치는 봉"으로 놓고 불합격을 기대했다.
    #    **기대가 틀렸다** — 4시그마는 깨끗이 분리된다(골 0.188). 통계량이 옳았다.
    #    진짜 겹치는 경우는 1.5시그마다.
    lo = list(rng.normal(39.0, 1.0, 400)); hi = list(rng.normal(40.5, 1.0, 400))
    v, *_ = valley_ratio(lo + hi)
    ok = v > G1_MAX_VALLEY
    print(f"[3b] 겹치는 두 봉(39/40.5 = 1.5시그마) 골 → {v:.3f}  게이트 초과여야  ", end="")
    print("OK" if ok else "!! 실패 — 붙어 있는 봉을 분리됐다고 한다"); bad += (not ok)

    lo = list(rng.normal(39.0, 1.0, 400)); hi = list(rng.normal(43.0, 1.0, 400))
    v, _, pk_hi = valley_ratio(lo + hi)
    ok = v <= G1_MAX_VALLEY
    print(f"[3c] 떨어진 두 봉(39/43 = 4시그마) 골 → {v:.3f}  게이트 이하여야  ", end="")
    print("OK" if ok else "!! 실패 — 분리된 봉을 못 잡는다"); bad += (not ok)
    print(f"     ⚠️ 단 봉우리 간격 4mm 는 그리퍼 가동폭으로 너무 작다. "
          f"G1 은 '움직였나'만 보고, 크기는 G3 가 본다")

    m = per_episode_min([[64, 50, 37.5], [66, 40, 38.0], [65, 45, 36.0]])
    ok = m == 37.5
    print(f"[4] 편별 최소의 중앙 → {m}  기대 37.5  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    ok = per_episode_min([]) is None and per_episode_min([[]]) is None
    print(f"[5] 빈 입력 → None (0 이 아니다)  ", end="")
    print("OK" if ok else "!! 실패 — 빈 것과 0mm 가 같은 출력이 된다"); bad += (not ok)

    d = abs(37.26 - 41.0)
    ok = d <= G3_MAX_DIFF_MM
    print(f"[6] 0911 기지값 대조 |37.26-41| = {d:.2f}mm <= {G3_MAX_DIFF_MM}  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    d = abs(37.26 - 20.0)
    ok = d > G3_MAX_DIFF_MM
    print(f"[7] 고의 오답(물체 20mm) |37.26-20| = {d:.2f}mm > {G3_MAX_DIFF_MM} → 불합격  ", end="")
    print("OK" if ok else "!! 실패 — 틀린 폭에도 통과한다"); bad += (not ok)

    ok = longest_run_false([True, False, False, False, True, False]) == 3
    print(f"[8] 최장 X 연속 → 3  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    # 단위 사고: m 로 들어와야 하는데 mm 로 들어온 경우
    ok = max([0.064, 0.039]) < 0.2 and max([64.0, 39.0]) > 0.2
    print(f"[9] 단위 판별 경계(0.2) 동작  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


# ── 판정 ─────────────────────────────────────────────────────────────────

def read_gripper(p: Path) -> tuple[list[float], list[bool], int]:
    """Return (detected gaps in mm, per-row detected flags, total rows)."""
    gaps: list[float] = []
    flags: list[bool] = []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = (r.get("status") or "").strip() == "D"
            flags.append(d)
            v = (r.get("gap_m") or "").strip()
            if d and v:
                gaps.append(float(v))
    return (gaps, flags, len(flags))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root")
    ap.add_argument("--object-width-mm", type=float, default=None,
                    help="그리퍼가 무는 방향의 물체 실측 폭 [mm]. 없으면 G3 판정 불가")
    ap.add_argument("--glob", default="gripper.csv")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 판정하지 않는다")
    print()

    root = Path(a.root).expanduser()
    files = sorted(root.rglob(a.glob))
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    print(f"대상 {len(files)} / 하위 디렉터리 {len(dirs)}  (루트 {root})")
    if not files:
        raise SystemExit(f"!! `{a.glob}` 0건. **없는 게 아니라 못 찾은 것일 수 있다** — "
                         "루트와 파일명을 먼저 확인하라")
    if dirs and len(files) < len(dirs):
        print(f"  ⚠️ 디렉터리 {len(dirs)}개 중 {len(files)}개만 `{a.glob}` 를 가진다")

    ep_gaps, rows, all_gaps = [], [], []
    for p in files:
        g, fl, tot = read_gripper(p)
        gmm = [x * 1000.0 for x in g]
        ep_gaps.append(gmm); all_gaps += gmm
        rows.append({"episode": p.parent.name, "rows": tot, "detected": len(g),
                     "rate": round(len(g) / tot, 4) if tot else 0.0,
                     "longest_x": longest_run_false(fl),
                     "min_mm": round(min(gmm), 3) if gmm else None,
                     "max_mm": round(max(gmm), 3) if gmm else None})

    det = sum(r["detected"] for r in rows); tot = sum(r["rows"] for r in rows)
    print(f"\n검출 {det} / 전체 행 {tot} = {det/tot:.4f}" if tot else "\n행 0")
    empty = [r["episode"] for r in rows if r["detected"] == 0]
    print(f"검출 0건 편: {len(empty)} / {len(rows)}" + (f"  {empty[:5]}" if empty else ""))

    unit = "m" if all_gaps and max(all_gaps) > 200 else "?"
    if all_gaps and max(all_gaps) > 200:
        print(f"\n⚠️ 최대 gap {max(all_gaps):.1f}mm — 입력이 이미 mm 였을 수 있다 (m 로 가정해 ×1000 함)")

    sep, ma, mb, pooled = separation(all_gaps)
    vr, pk_lo, pk_hi = valley_ratio(all_gaps)
    g1 = vr <= G1_MAX_VALLEY
    print(f"\nG1 이봉성   골 비율 {vr:.3f}  (게이트 <= {G1_MAX_VALLEY})  "
          f"봉우리 {pk_lo:.2f}mm / {pk_hi:.2f}mm  → {'통과' if g1 else '불합격'}")
    print(f"            [진단] 봉우리 간격 {abs(pk_hi-pk_lo):.2f}mm · Otsu 분리도 {sep:.2f} · "
          f"저군집 {ma:.2f}mm · 고군집 {mb:.2f}mm · 합산표준편차 {pooled:.2f}")
    print(f"            ⚠️ 봉우리 간격에는 게이트를 두지 않았다. 분포를 본 적이 없다 — "
          f"트랙 A 산출물 모이면 정한다")

    g2 = per_episode_min(ep_gaps)
    print(f"G2 폐쇄측   편별 최소 gap 의 중앙 {g2:.3f} mm  (n={sum(1 for g in ep_gaps if g)}/{len(ep_gaps)})"
          if g2 is not None else "G2 폐쇄측   판정 불가 — 검출된 행이 없다")

    if a.object_width_mm is None:
        g3, g3txt = None, ("G3 대조     **판정 불가** — --object-width-mm 미제공. "
                           "통과가 아니다. 물체 폭을 재서 다시 돌려라")
    elif g2 is None:
        g3, g3txt = None, "G3 대조     판정 불가 — G2 가 없다"
    else:
        d = abs(g2 - a.object_width_mm)
        g3 = d <= G3_MAX_DIFF_MM
        g3txt = (f"G3 대조     |{g2:.2f} - {a.object_width_mm:.2f}| = {d:.2f} mm  "
                 f"(허용 {G3_MAX_DIFF_MM})  → {'통과' if g3 else '불합격'}")
    print(g3txt)

    verdict = "사용 가능" if (g1 and g3) else ("판정 불가" if g3 is None else "학습에 넣지 않는다")
    print(f"\n판정: gap 채널 — **{verdict}**")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"root": str(root), "episodes": len(rows), "detected": det, "rows_total": tot,
             "object_width_mm": a.object_width_mm, "unit_guess": unit,
             "G1": {"valley_ratio": round(vr, 4), "gate": G1_MAX_VALLEY, "pass": g1,
                    "peak_low_mm": round(pk_lo, 3), "peak_high_mm": round(pk_hi, 3),
                    "peak_gap_mm": round(abs(pk_hi - pk_lo), 3),
                    "otsu_separation": round(sep, 4),
                    "low_mean_mm": round(ma, 3), "high_mean_mm": round(mb, 3)},
             "G2": {"median_min_gap_mm": round(g2, 3) if g2 is not None else None},
             "G3": {"gate_mm": G3_MAX_DIFF_MM, "pass": g3},
             "verdict": verdict, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
