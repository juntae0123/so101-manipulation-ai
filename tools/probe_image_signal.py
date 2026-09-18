#!/usr/bin/env python3
"""Measure whether dataset images carry any signal at all.
데이터셋 이미지에 신호가 있기는 한지 계측한다 — 경로 고장인지, 볼 게 없는지 가른다.

noise96 어블레이션에서 이미지를 96계조 노이즈로 뭉개도 val 이 그대로였다 (3시드).
정책이 이미지를 안 쓴다는 뜻인데 원인이 안 갈렸다:
  (1) 이미지 경로 고장 — 전부 0이거나 상수 프레임
  (2) 데이터에 볼 게 없음 — 시연마다 물체 위치가 안 변했다
  (3) 인코더·정규화 문제
이 도구는 (1)과 (2)를 가른다. 모든 수치는 0~255 계조, 주입 노이즈 96과 같은 자다.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

INJECTED_NOISE_STD = 96.0


def find_image_arrays(npz):
    """Return arrays that look like image stacks.
    이미지 스택으로 보이는 배열만 골라 돌려준다."""
    out = {}
    for k in npz.files:
        a = npz[k]
        if a.ndim >= 3 and a.dtype == np.uint8:
            out[k] = a
        elif a.ndim >= 4 and 224 in a.shape:
            out[k] = a
    return out


def as_tchw(a):
    """Collapse a (T, cam, C, H, W) stack to (T, C, H, W) using camera 0.
    카메라 축이 있으면 0번만 쓴다."""
    if a.ndim == 5:
        return a[:, 0]
    return a


def describe(path):
    """Print keys and shapes of one episode.
    에피소드 하나의 키와 모양을 찍는다."""
    with np.load(path, allow_pickle=True) as z:
        print(f"  [{path.name}]")
        for k in z.files:
            a = z[k]
            print(f"    {k:<20} shape={str(a.shape):<24} dtype={a.dtype}")


def main():
    """Entry point.
    진입점."""
    ap = argparse.ArgumentParser(
        description="데이터셋 이미지에 신호가 있는지 계측한다 (경로 고장 vs 장면 불변)",
    )
    ap.add_argument("--data", required=True, help="계약 에피소드 디렉터리")
    ap.add_argument("--episodes", type=int, default=20, help="표본으로 쓸 에피소드 수")
    ap.add_argument("--pairs", type=int, default=30, help="에피소드 간 비교 쌍 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="결과 JSON 경로")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.data)
    files = sorted(root.glob("*.npz")) or sorted(root.rglob("*.npz"))
    if not files:
        print(f"!! .npz 를 못 찾았다: {root}")
        return 1

    print(f"\n에피소드 {len(files)}편 발견 · {root}")
    print("\n[ 0. 첫 에피소드 구조 ]")
    describe(files[0])

    with np.load(files[0], allow_pickle=True) as z:
        cand = find_image_arrays(z)
    if not cand:
        print("\n!! 이미지로 보이는 배열이 없다. 위 구조를 보고 도구를 고쳐야 한다.")
        return 1
    img_key = sorted(cand, key=lambda k: -cand[k].size)[0]
    print(f"\n이미지 배열로 고른 키: {img_key}")

    sample = files[: args.episodes] if len(files) > args.episodes else files

    within, first_frames, pix_std, all_zero = [], [], [], 0
    for f in sample:
        with np.load(f, allow_pickle=True) as z:
            a = as_tchw(z[img_key]).astype(np.float32)
        if a.shape[0] < 2:
            continue
        if float(a.max()) == 0.0:
            all_zero += 1
        pix_std.append(float(a.std()))
        step = max(1, a.shape[0] // 20)
        idx = list(range(0, a.shape[0] - 1, step))
        within.append(float(np.mean([np.abs(a[i] - a[i + 1]).mean() for i in idx])))
        first_frames.append(a[0])

    between = []
    n = len(first_frames)
    for _ in range(args.pairs):
        if n < 2:
            break
        i, j = rng.sample(range(n), 2)
        between.append(float(np.abs(first_frames[i] - first_frames[j]).mean()))

    w_med = float(np.median(within)) if within else 0.0
    b_med = float(np.median(between)) if between else 0.0
    s_med = float(np.median(pix_std)) if pix_std else 0.0

    print(f"\n[ 1. 계측 ] 표본 {len(sample)}편 · 단위 0~255 계조")
    print(f"  픽셀 표준편차 (중앙)        {s_med:8.2f}")
    print(f"  프레임 간 차이 (중앙)       {w_med:8.2f}   <- 0 이면 경로 고장")
    print(f"  에피소드 간 첫프레임 차이   {b_med:8.2f}   <- 0 이면 장면이 매번 같다")
    print(f"  학습에 주입한 노이즈 std    {INJECTED_NOISE_STD:8.2f}   <- 같은 자")
    if all_zero:
        print(f"  !! 전부 0인 에피소드 {all_zero}편")

    print("\n[ 2. 판정 ]")
    if all_zero == len(sample) or s_med < 1.0:
        verdict = "(1) 이미지 경로 고장 — 값이 사실상 비어 있다"
    elif w_med < 1.0:
        verdict = "(1) 경로 고장 — 같은 프레임이 반복된다"
    elif b_med < 1.0:
        verdict = "(2) 장면이 매번 같다 — 모델이 이미지를 볼 이유가 없다"
    elif b_med < INJECTED_NOISE_STD:
        verdict = (
            f"(2) 에 가깝다 — 에피소드 간 신호({b_med:.1f})가 "
            f"주입 노이즈({INJECTED_NOISE_STD:.0f})보다 작다. 신호가 묻힌다"
        )
    else:
        verdict = "(3) 으로 넘어간다 — 데이터에는 신호가 있다. 인코더·정규화를 본다"
    print(f"  {verdict}")

    print("\n  주의 — 이 도구가 답하지 않는 것")
    print("     - 그 차이가 물체 위치 때문인지 조명·손떨림 때문인지 구분하지 않는다")
    print("     - 정책이 그 신호를 쓸 수 있는지는 별개다 (3)")

    result = {
        "data": str(root),
        "n_episodes_total": len(files),
        "n_sampled": len(sample),
        "image_key": img_key,
        "pixel_std_median": s_med,
        "within_episode_diff_median": w_med,
        "between_episode_diff_median": b_med,
        "injected_noise_std": INJECTED_NOISE_STD,
        "all_zero_episodes": all_zero,
        "verdict": verdict,
        "unit": "0-255 gray levels",
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n결과: {args.out}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
