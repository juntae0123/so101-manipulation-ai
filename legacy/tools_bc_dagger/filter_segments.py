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
    ap.add_argument("--min-steps", type=int, default=0,
                    help="이 틱 수 미만인 에피소드를 버린다. "
                         "--chunk 와 함께 쓰면 둘 중 큰 쪽이 기준이 된다")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 이 아니면 이 틱 수를 넘는 에피소드도 버린다")
    # Valid anchors, not raw length, is what decides whether a segment can be used.
    # 조각을 쓸 수 있는지 정하는 것은 길이가 아니라 **유효 anchor 수** 다.
    #
    # 유효 anchor 수 = L - H - K + 1
    #   L = 조각 길이(틱) · H = 관측 history 스텝 수 · K = 행동 청크 길이
    #
    # ⚠️ H 는 경로마다 다르다. 계약 0.3.0 은 단일 프레임이라 H=1,
    # 트랙 A 의 `umi_relative_chunk` 는 history 2스텝이라 H=2 다 (0915 합의).
    # 기본값을 1 로 두되 인자로 받는 이유가 이것이다 — 한쪽 값을 코드에 박으면
    # 다른 경로가 조용히 틀린 기준으로 걸러진다.
    ap.add_argument("--chunk", type=int, default=0,
                    help="행동 청크 길이 K. 주면 유효 anchor >= 1 을 기준으로 거른다")
    ap.add_argument("--obs-history", type=int, default=1,
                    help="관측 history 스텝 수 H. 계약 0.3.0 은 1, v6 는 2")
    args = ap.parse_args()

    if args.chunk < 0 or args.obs_history < 1:
        raise SystemExit("--chunk 는 0 이상, --obs-history 는 1 이상이어야 한다")
    need = args.min_steps
    if args.chunk:
        need = max(need, args.obs_history + args.chunk)
    if need <= 0:
        raise SystemExit("--min-steps 또는 --chunk 중 하나는 줘야 한다")

    npzs = sorted(args.src.glob("*.npz"))
    if not npzs:
        print(f"!! .npz 없음: {args.src}")
        return 1

    lengths = {f: episode_steps(f) for f in npzs}
    keep = [f for f, n in lengths.items()
            if n >= need and (args.max_steps == 0 or n <= args.max_steps)]
    drop = [f for f in npzs if f not in keep]

    def anchors(n: int) -> int:
        """Usable anchors in a segment of n ticks, clamp/padding forbidden.
        n 틱 조각에서 쓸 수 있는 anchor 수. 클램프·패딩을 쓰지 않는다는 전제다."""
        if not args.chunk:
            return n
        return max(0, n - args.obs_history - args.chunk + 1)

    kept_ticks = sum(lengths[f] for f in keep)
    drop_ticks = sum(lengths[f] for f in drop)
    kept_anchors = sum(anchors(lengths[f]) for f in keep)

    # 전체·채택·폐기를 전부 찍는다. 부분만 찍으면 어디로 샜는지 알 수 없다.
    print(f"{args.src}")
    print(f"  기준  길이 >= {need}"
          + (f"  (H={args.obs_history} + K={args.chunk})" if args.chunk else "")
          + (f", <= {args.max_steps}" if args.max_steps else ""))
    print(f"  세그먼트  전체 {len(npzs)} = 채택 {len(keep)} + 폐기 {len(drop)}")
    print(f"  틱        전체 {kept_ticks + drop_ticks} = 채택 {kept_ticks} "
          f"+ 폐기 {drop_ticks}")
    if args.chunk:
        print(f"  유효 anchor  채택분 합계 {kept_anchors}"
              f" (클램프·패딩 사용: false)")
    if drop:
        dl = sorted(lengths[f] for f in drop)
        print(f"  폐기 길이  최소 {dl[0]} · 중앙 {dl[len(dl) // 2]} · 최대 {dl[-1]}")
    if len(keep) + len(drop) != len(npzs):
        print("!! 세그먼트 합이 안 맞는다")
        return 1
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
