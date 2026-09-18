#!/usr/bin/env python3
"""Merge contract datasets (and their sidecars) into one directory.
계약 데이터셋과 사이드카를 한 디렉터리로 합친다.

DAgger 원 방식은 전문가 데이터와 교정 데이터를 **섞어서** 학습한다. 교정 데이터만
쓰면 정책이 "실패 상태에서 회복하는 법"만 배우고 처음부터 제대로 가는 법을 배우지
못한다. 2026-09-15 에 그 반쪽 구현으로 0/300 을 받았다.

파일명이 겹치므로 소스별 접두어를 붙인다. 사이드카(`*.command.npy`)도 같이 옮긴다 —
빠뜨리면 학습 타깃이 계약 action 으로 조용히 바뀐다.

    # [서버]
    python tools/merge_datasets.py --out datasets/mix_cmd \
        --src datasets/sim_pick_cmd --src datasets/dagger_cmd
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    """Entry point.
    진입점."""
    ap = argparse.ArgumentParser(description="계약 데이터셋 병합 (사이드카 포함)")
    ap.add_argument("--src", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    total_npz = 0
    total_side = 0
    for src in args.src:
        npzs = sorted(src.glob("*.npz"))
        if not npzs:
            print(f"!! .npz 없음: {src}")
            return 1
        prefix = src.name
        n_side = 0
        for f in npzs:
            stem = f"{prefix}__{f.stem}"
            shutil.copy2(f, args.out / f"{stem}.npz")
            meta = f.with_suffix(".json")
            if meta.exists():
                shutil.copy2(meta, args.out / f"{stem}.json")
            # 사이드카. 빠뜨리면 학습 타깃이 조용히 계약 action 으로 바뀐다.
            for side in src.glob(f"{f.stem}.*.npy"):
                suffix = side.name[len(f.stem) + 1:]
                shutil.copy2(side, args.out / f"{stem}.{suffix}")
                n_side += 1
        print(f"  {src.name}: npz {len(npzs)} · 사이드카 {n_side}")
        total_npz += len(npzs)
        total_side += n_side

    out_npz = len(list(args.out.glob("*.npz")))
    out_side = len(list(args.out.glob("*.command.npy")))
    print(f"\n{args.out}: npz {out_npz} · command 사이드카 {out_side}")
    if out_npz != total_npz:
        print(f"!! npz 개수가 안 맞는다: {out_npz} != {total_npz}")
        return 1
    if out_side != out_npz:
        print(f"!! 사이드카가 npz 와 개수가 다르다: {out_side} != {out_npz}")
        return 1
    print("병합 완료 — 개수 검증 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
