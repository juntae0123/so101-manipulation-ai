#!/usr/bin/env python3
"""Measure how much of the action is explained without looking at anything.
행동이 관측 없이 얼마나 설명되는지 잰다 — "이미지가 필요 없는 문제인가"를 가른다.

계약 0.3.0 에서 action[t] = state[t+1] 이다. UMI 시연을 IK 로 관절공간에 옮기면
매끄러운 1차원 궤적이 되고, state 는 그 궤적 위 위치를 이미 알려준다.
그러면 다음 상태가 "가던 방향으로 계속" 으로 대부분 풀린다 — 이미지가 기여할 여지가 없다.

주의: 이 도구는 train_bc 의 val loss 를 재현하지 않는다. 그쪽은 관절별 표준화 +
그리퍼 이진분류가 섞인 단위라 밖에서 다시 구현하면 틀릴 위험이 크다.
여기서는 단위 없는 비율만 보고 판정한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EXPLAINED_GATE = 0.70
SAME_TRAJ_GATE = 0.20
RESAMPLE = 100


def resample(a, n):
    """Resample a (T, D) trajectory to (n, D) on a normalised time axis.
    (T, D) 궤적을 정규화 시간축 위 (n, D) 로 다시 뽑는다."""
    t_old = np.linspace(0.0, 1.0, a.shape[0])
    t_new = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t_new, t_old, a[:, d]) for d in range(a.shape[1])], axis=1)


def main():
    """Entry point.
    진입점."""
    ap = argparse.ArgumentParser(
        description="행동이 관측 없이 얼마나 설명되는지 잰다",
    )
    ap.add_argument("--data", required=True, help="계약 에피소드 디렉터리")
    ap.add_argument("--episodes", type=int, default=0, help="0 이면 전부")
    ap.add_argument("--out", default="", help="결과 JSON 경로")
    args = ap.parse_args()

    root = Path(args.data)
    files = sorted(root.glob("*.npz")) or sorted(root.rglob("*.npz"))
    if not files:
        print(f"!! .npz 를 못 찾았다: {root}")
        return 1
    if args.episodes:
        files = files[: args.episodes]

    ident, extrap, trajs = [], [], []
    grip_flips, n_steps = [], 0
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            if "state" not in z.files or "action" not in z.files:
                print("!! state/action 키가 없다.")
                return 1
            s = z["state"].astype(np.float64)
            a = z["action"].astype(np.float64)
        if s.shape[0] < 3:
            continue
        n_steps += s.shape[0]
        ident.append(np.abs(a[1:-1] - s[1:-1]).mean(axis=0))
        pred = 2.0 * s[1:-1] - s[:-2]
        extrap.append(np.abs(a[1:-1] - pred).mean(axis=0))
        trajs.append(resample(s, RESAMPLE))
        g = s[:, 5]
        grip_flips.append(int((np.abs(np.diff(np.sign(g - g.mean()))) > 0).sum()))

    ident_m = np.median(np.stack(ident), axis=0)
    extrap_m = np.median(np.stack(extrap), axis=0)
    explained = 1.0 - np.divide(extrap_m, ident_m, out=np.zeros_like(extrap_m), where=ident_m > 0)

    print(f"\n에피소드 {len(files)}편 · 스텝 {n_steps} · 단위 계약 [-1,1]")
    print("\n[ 1. 관절별 — 중앙값 ]")
    print(f"  {'관절':<16}{'A 델타크기':>12}{'B 외삽잔차':>12}{'C 설명력':>10}")
    for i, j in enumerate(JOINTS):
        print(f"  {j:<16}{ident_m[i]:>12.5f}{extrap_m[i]:>12.5f}{explained[i]:>9.1%}")

    arm = slice(0, 5)
    arm_expl = 1.0 - extrap_m[arm].sum() / max(ident_m[arm].sum(), 1e-12)
    print(f"\n  팔 5축 합산 설명력  {arm_expl:>8.1%}")

    T = np.stack(trajs)
    between = np.abs(T[:, None] - T[None, :]).mean(axis=(2, 3))
    iu = np.triu_indices(len(T), k=1)
    between_m = float(np.median(between[iu])) if len(iu[0]) else 0.0
    within_m = float(np.median(T.max(axis=1) - T.min(axis=1)))
    ratio = between_m / max(within_m, 1e-12)

    print("\n[ 2. 에피소드 간 궤적 ]")
    print(f"  에피소드 간 차이 (중앙)   {between_m:8.5f}")
    print(f"  에피소드 내 변화폭 (중앙) {within_m:8.5f}")
    print(f"  비율                      {ratio:8.1%}   <- 작을수록 '한 궤적의 반복'")
    print(f"  그리퍼 전환 횟수 (중앙)   {np.median(grip_flips):8.0f}")

    print("\n[ 3. 판정 — 사전등록 기준 ]")
    v1 = arm_expl >= EXPLAINED_GATE
    v2 = ratio < SAME_TRAJ_GATE
    print(f"  외삽 설명력 {arm_expl:.1%} {'>=' if v1 else '<'} {EXPLAINED_GATE:.0%}"
          f"  ->  {'움직임 대부분이 외삽으로 설명된다' if v1 else '외삽만으로는 부족하다'}")
    print(f"  궤적 비율   {ratio:.1%} {'<' if v2 else '>='} {SAME_TRAJ_GATE:.0%}"
          f"  ->  {'사실상 한 궤적이다. BC = replay' if v2 else '에피소드마다 궤적이 다르다'}")

    if v1 and v2:
        verdict = "가설 확정 — 이미지 없이 풀리는 문제이고, 데이터는 한 궤적의 반복이다"
    elif v1:
        verdict = "외삽으로 대부분 풀린다. 다만 궤적은 에피소드마다 다르다"
    elif v2:
        verdict = "궤적은 비슷하나 외삽만으로는 부족하다 — 관측이 필요한 구간이 있다"
    else:
        verdict = "가설 기각 — 관측이 필요한 문제다"
    print(f"\n  {verdict}")

    print("\n  이 도구가 답하지 않는 것")
    print("     - train_bc 의 val loss 와 단위가 다르다. 직접 비교하지 마라")
    print("     - 물체가 움직였을 때 어떻게 되는지는 롤아웃에서만 나온다")

    result = {
        "data": str(root), "n_episodes": len(files), "n_steps": n_steps,
        "joints": JOINTS,
        "identity_median": ident_m.tolist(),
        "extrapolation_residual_median": extrap_m.tolist(),
        "explained_per_joint": explained.tolist(),
        "arm_explained": float(arm_expl),
        "between_episode_traj_median": between_m,
        "within_episode_range_median": within_m,
        "between_over_within": float(ratio),
        "gates": {"explained": EXPLAINED_GATE, "same_traj": SAME_TRAJ_GATE},
        "verdict": verdict,
        "unit": "contract [-1,1]",
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n결과: {args.out}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
