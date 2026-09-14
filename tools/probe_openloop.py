#!/usr/bin/env python3
"""Open-loop N-step prediction error of observation-free baselines.
관측 없이 굴리는 기준선의 오픈루프 N스텝 예측 오차를 잰다.

실데이터로 학습한 정책을 평가할 폐루프 경로가 없다 (이슈 166).
그런데 오픈루프 예측 오차는 데이터만으로 잴 수 있다.
  identity      s[t+1] = s[t]
  extrapolation s[t+1] = s[t] + (s[t] - s[t-1])
학습 정책은 이 선을 넘어야 "관측을 쓴다"고 말할 수 있다.
stride 를 흔들어 스텝당 예측 난이도가 올라가는지도 같이 본다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARM = slice(0, 5)
HORIZONS_S = [0.033, 0.17, 0.5, 1.0, 2.0, 3.0]
FPS = 30.0


def rollout_error(s, horizons, mode):
    """Open-loop error at each horizon for one episode.
    에피소드 하나에 대해 지평별 오픈루프 오차를 낸다."""
    T = s.shape[0]
    out = {h: [] for h in horizons}
    hmax = max(horizons)
    step = max(1, (T - hmax) // 12) if T > hmax else 1
    for t0 in range(1, max(2, T - hmax), step):
        if t0 + hmax >= T:
            break
        cur = s[t0].copy()
        vel = s[t0] - s[t0 - 1]
        for h in range(1, hmax + 1):
            if mode == "extrap":
                cur = cur + vel
            if h in out:
                out[h].append(float(np.abs(cur[ARM] - s[t0 + h][ARM]).mean()))
    return out


def main():
    """Entry point.
    진입점."""
    ap = argparse.ArgumentParser(description="관측 없는 기준선의 오픈루프 오차 + stride 스윕")
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    results = {}
    for d in args.data:
        root = Path(d)
        files = sorted(root.glob("*.npz")) or sorted(root.rglob("*.npz"))
        if not files:
            print(f"!! .npz 없음: {root}")
            continue
        if args.episodes:
            files = files[: args.episodes]
        states = []
        for f in files:
            with np.load(f, allow_pickle=True) as z:
                if "state" not in z.files:
                    print(f"!! state 키 없음: {f}")
                    return 1
                states.append(z["state"].astype(np.float64))

        rng_med = float(np.median([np.abs(s[:, ARM].max(0) - s[:, ARM].min(0)).mean() for s in states]))
        print(f"\n{'='*64}\n{root}  ({len(files)}편)")
        print(f"  에피소드 내 팔 변화폭 (중앙) {rng_med:.5f}   <- 오차를 이것과 견준다")

        print(f"\n  [ 스텝당 — 외삽 설명력 ]")
        print(f"    {'stride':>8}{'Hz':>7}{'델타크기':>12}{'외삽잔차':>12}{'설명력':>10}")
        expl_by_stride = {}
        for st in args.strides:
            ids, exs = [], []
            for s in states:
                ss = s[::st]
                if ss.shape[0] < 4:
                    continue
                a = ss[1:]
                cur = ss[1:-1]
                ids.append(np.abs(a[1:] - cur).mean(axis=0))
                pred = 2.0 * ss[1:-1] - ss[:-2]
                exs.append(np.abs(a[1:] - pred).mean(axis=0))
            i_m = np.median(np.stack(ids), axis=0)
            e_m = np.median(np.stack(exs), axis=0)
            expl = 1.0 - e_m[ARM].sum() / max(i_m[ARM].sum(), 1e-12)
            expl_by_stride[st] = float(expl)
            print(f"    {st:>8}{FPS/st:>7.0f}{i_m[ARM].mean():>12.5f}{e_m[ARM].mean():>12.5f}{expl:>9.1%}")

        per_stride = {}
        for st in args.strides:
            hz = FPS / st
            hor = sorted({max(1, int(round(sec * hz))) for sec in HORIZONS_S})
            acc = {"identity": {h: [] for h in hor}, "extrap": {h: [] for h in hor}}
            for s in states:
                ss = s[::st]
                if ss.shape[0] < max(hor) + 3:
                    continue
                for mode in ("identity", "extrap"):
                    e = rollout_error(ss, hor, mode)
                    for h in hor:
                        acc[mode][h].extend(e[h])
            print(f"\n  [ stride {st}  =  {hz:.0f}Hz ]")
            print(f"    {'지평':>8}{'스텝':>7}{'identity':>12}{'외삽':>12}{'외삽/변화폭':>13}")
            rows = []
            for sec in HORIZONS_S:
                h = max(1, int(round(sec * hz)))
                if h not in acc["extrap"] or not acc["extrap"][h]:
                    continue
                i_v = float(np.median(acc["identity"][h]))
                e_v = float(np.median(acc["extrap"][h]))
                print(f"    {sec:>7.2f}s{h:>7d}{i_v:>12.5f}{e_v:>12.5f}{e_v/max(rng_med,1e-12):>12.1%}")
                rows.append({"sec": sec, "steps": h, "identity": i_v, "extrap": e_v})
            per_stride[st] = rows

        results[str(root)] = {"range_median": rng_med,
                              "explained_by_stride": expl_by_stride,
                              "per_stride": per_stride}

    print(f"\n{'='*64}\n[ 판정 — 사전등록 기준 ]")
    print("  재는 것: 스텝당 외삽 설명력. 낮아야 모델이 관측을 볼 이유가 생긴다.")
    print("  게이트:  stride 를 끝까지 키웠을 때 설명력이 15%p 이상 떨어지는가")
    print("  (오픈루프 표는 게이트가 아니라 학습 정책이 넘어야 할 기준선이다)")
    for name, r in results.items():
        e = r["explained_by_stride"]
        if not e:
            continue
        ks = sorted(e)
        drop = e[ks[0]] - e[ks[-1]]
        verdict = ("다운샘플이 외삽 지름길을 막는다 — 처방 후보로 유효"
                   if drop >= 0.15 else
                   "다운샘플로는 외삽이 안 막힌다 — 처방에서 뺀다")
        chain = "  ->  ".join(f"stride{k} {e[k]:.1%}" for k in ks)
        print(f"\n  {Path(name).name}")
        print(f"    {chain}")
        print(f"    낙폭 {drop*100:.1f}%p  ->  {verdict}")

    print("\n  이 도구가 답하지 않는 것")
    print("     - 학습 정책이 이 선을 넘는지는 ckpt 를 굴려야 안다")
    print("     - 오픈루프 오차가 작아도 접촉·파지는 별개다")
    print("     - train_bc 의 val loss 와 단위가 다르다\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"결과: {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
