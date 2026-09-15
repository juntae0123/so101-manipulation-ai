#!/usr/bin/env python3
"""Filter a dataset by episode length, keeping sidecars in step.
에피소드 길이로 데이터셋을 거른다. 사이드카를 같이 옮긴다.

왜 있나 — 2026-09-15 계측 🟢. DAgger 세그먼트 120편의 길이 분포가 **이봉**이다.

    길이  2: 19편   4: 26편   8: 10편   16: 4편   32: 3편   64+: 58편
    중앙 62 · 평균 83.6 · 최대 170 (에피소드 전체 길이)

청크 K=8 로 학습할 때 길이 2 짜리 조각은 `data/dataset.py` 의 종단 클램프가
마지막 상태를 6번 반복해 채운다. **그 샘플이 가르치는 것은 "가만히 있기"다.**
K 보다 짧은 조각이 전체의 38% 이상(45편)이다.

이 도구는 그 조각을 버린 데이터셋을 만든다. 버리는 게 나은지는 롤아웃이 답한다 —
이 도구는 가설을 검증하지 않고 **검증할 데이터를 만든다.**

⚠️ 사이드카(`*.command.npy`)를 빠뜨리면 학습 타깃이 계약 action 으로 조용히
   바뀐다. merge_datasets.py 와 같은 이유로 개수를 검증한다.

    # [서버]
    python tools/filter_segments.py --src datasets/dagger_cmd \
        --out datasets/dagger_cmd_min8 --min-steps 8
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def episode_steps(npz: Path) -> int:
    """Number of ticks in a contract episode.
    계약 에피소드의 틱 수.

    메타 json 의 `n_steps` 가 아니라 배열에서 직접 읽는다 — 메타와 배열이 갈릴 수
    있고, 학습이 보는 것은 배열이다."""
    with np.load(npz, allow_pickle=True) as z:
        return int(z["state"].shape[0])


def main() -> int:
    """Entry point.
    진입점."""
    ap = argparse.ArgumentParser(description="에피소드 길이 필터 (사이드카 포함)")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-steps", type=int, required=True,
                    help="이 틱 수 미만인 에피소드를 버린다")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 이 아니면 이 틱 수를 넘는 에피소드도 버린다")
    args = ap.parse_args()

    npzs = sorted(args.src.glob("*.npz"))
    if not npzs:
        print(f"!! .npz 없음: {args.src}")
        return 1

    lengths = {f: episode_steps(f) for f in npzs}
    keep = [f for f, n in lengths.items()
            if n >= args.min_steps and (args.max_steps == 0 or n <= args.max_steps)]
    drop = [f for f in npzs if f not in keep]

    print(f"{args.src}: {len(npzs)}편")
    print(f"  유지 {len(keep)}편 · 버림 {len(drop)}편 "
          f"(기준 {args.min_steps} 이상"
          + (f", {args.max_steps} 이하" if args.max_steps else "") + ")")
    if drop:
        dl = sorted(lengths[f] for f in drop)
        print(f"  버린 길이: 최소 {dl[0]} · 중앙 {dl[len(dl) // 2]} · 최대 {dl[-1]}")
    if not keep:
        print("!! 남는 에피소드가 없다. 기준을 다시 보라")
        return 1

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    n_side = 0
    for f in keep:
        shutil.copy2(f, args.out / f.name)
        meta = f.with_suffix(".json")
        if meta.exists():
            shutil.copy2(meta, args.out / meta.name)
        for side in args.src.glob(f"{f.stem}.*.npy"):
            shutil.copy2(side, args.out / side.name)
            n_side += 1

    out_npz = len(list(args.out.glob("*.npz")))
    out_side = len(list(args.out.glob("*.command.npy")))
    print(f"\n{args.out}: npz {out_npz} · command 사이드카 {out_side}")
    # 개수 검증. 조용한 불일치가 타깃을 바꾼다
    if out_npz != len(keep):
        print(f"!! npz 개수가 안 맞는다: {out_npz} != {len(keep)}")
        return 1
    if out_side != out_npz:
        print(f"!! 사이드카가 npz 와 개수가 다르다: {out_side} != {out_npz}")
        return 1
    print("필터 완료 — 개수 검증 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
