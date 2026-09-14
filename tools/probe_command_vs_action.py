#!/usr/bin/env python3
"""Compare two action definitions by how predictable they are without observations.
두 액션 정의를 "관측 없이 얼마나 예측되는가"로 비교한다.

계약 0.3.0 은 action[t] = state[t+1] — 이미 간 거리.
옛 규약은 action = data.ctrl — 가려는 곳. 제어기가 쫓는 목표라 현재 자세보다 앞선다.

가설: ctrl 쪽이 외삽으로 덜 풀린다. 그러면 관측이 필요한 정보가 거기 있다는 뜻이다.
게이트: ctrl 설명력이 state[t+1] 기반보다 20%p 이상 낮으면 전제 확인.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARM = slice(0, 5)
GATE_DROP = 0.20


def explained(state, target):
    """Extrapolation explainability of a target given state history.
    state 이력만으로 타깃이 얼마나 설명되는지."""
    ids, exs = [], []
    for s, a in zip(state, target):
        if s.shape[0] < 4:
            continue
        cur = s[1:-1]
        tgt = a[1:-1]
        pred = 2.0 * s[1:-1] - s[:-2]
        ids.append(np.abs(tgt - cur).mean(axis=0))
        exs.append(np.abs(tgt - pred).mean(axis=0))
    i_m = np.median(np.stack(ids), axis=0)
    e_m = np.median(np.stack(exs), axis=0)
    expl = 1.0 - e_m[ARM].sum() / max(i_m[ARM].sum(), 1e-12)
    return float(expl), float(i_m[ARM].mean()), float(e_m[ARM].mean())


def main():
    """Entry point.
    진입점."""
    ap = argparse.ArgumentParser(description="계약 action vs 사이드카 command 비교")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.data)
    npzs = sorted(root.glob("*.npz"))
    if not npzs:
        print(f"!! .npz 없음: {root}")
        return 1

    states, actions, commands, miss = [], [], [], 0
    for f in npzs:
        c = f.with_suffix(".command.npy")
        if not c.exists():
            miss += 1
            continue
        with np.load(f, allow_pickle=True) as z:
            s = z["state"].astype(np.float64)
            a = z["action"].astype(np.float64)
        cmd = np.load(c).astype(np.float64)
        if cmd.shape != s.shape:
            print(f"!! shape 불일치 {f.name}: state{s.shape} command{cmd.shape}")
            return 1
        states.append(s); actions.append(a); commands.append(cmd)

    if not states:
        print(f"!! 사이드카 .command.npy 가 하나도 없다 (누락 {miss})")
        return 1

    ea, ia, xa = explained(states, actions)
    ec, ic, xc = explained(states, commands)
    lead = float(np.median([np.abs(c[1:-1] - s[1:-1]).mean() for s, c in zip(states, commands)]))
    move = float(np.median([np.abs(a[1:-1] - s[1:-1]).mean() for s, a in zip(states, actions)]))

    print(f"\n에피소드 {len(states)}편 (사이드카 누락 {miss}) · 단위 계약 [-1,1]")
    print(f"\n  {'타깃':<26}{'크기':>12}{'외삽잔차':>12}{'설명력':>10}")
    print(f"  {'action = state[t+1]':<26}{ia:>12.5f}{xa:>12.5f}{ea:>9.1%}")
    print(f"  {'command = ctrl':<26}{ic:>12.5f}{xc:>12.5f}{ec:>9.1%}")
    print(f"\n  명령 선행량 |ctrl - state|   {lead:.5f}")
    print(f"  실제 이동량 |state[t+1]-state| {move:.5f}")
    print(f"  선행/이동 비율               {lead / max(move, 1e-12):.2f}x")

    drop = ea - ec
    ok = drop >= GATE_DROP
    print(f"\n[ 판정 — 사전등록 게이트: ctrl 설명력이 {GATE_DROP:.0%} 이상 낮은가 ]")
    print(f"  낙폭 {drop * 100:.1f}%p  ->  "
          f"{'전제 확인 — ctrl 에 관측이 필요한 정보가 있다' if ok else '가설 기각 — 두 정의가 같은 난이도다'}")
    print("\n  이 도구가 답하지 않는 것")
    print("     - ctrl 로 학습하면 롤아웃이 나오는지는 학습해야 안다")
    print("     - 실물에 ctrl 에 해당하는 값이 있는지는 트랙 A 확인 사항이다\n")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "n": len(states), "sidecar_missing": miss,
            "action_explained": ea, "command_explained": ec, "drop": drop,
            "action_mag": ia, "command_mag": ic,
            "lead": lead, "move": move, "gate_drop": GATE_DROP, "passed": ok,
        }, ensure_ascii=False, indent=2))
        print(f"결과: {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
