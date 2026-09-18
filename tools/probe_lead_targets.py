#!/usr/bin/env python3
"""Data-level sweep: how predictable is each lead-k target without observations?
데이터 수준 스윕: 리드-k 타깃이 관측 없이 얼마나 예측되는가?

학습 없이 수 분에 끝난다. 여기서 이미 갈리면 학습을 안 돌려도 된다.

`probe_command_vs_action.py` 가 두 타깃을 비교했다면 이것은 k 를 훑는다.
게이트는 그 도구와 같은 정의를 쓴다 — `ctrl` 이 만든 낙폭(20.4%p)이 기준점이다.

⚠️ 이 도구는 **설명력만** 잰다. 설명력이 낮다고 학습이 잘 된다는 보장은 없다.
   리드가 너무 크면 타깃이 관측과 무관해져서 설명력도 낮고 학습도 안 될 수 있다.
   그 구분은 롤아웃으로만 된다.

    # [서버]
    python tools/probe_lead_targets.py --data datasets/sim_pick_cmd \
        --k 1 2 4 8 16 --command --out out/lead_sweep_0914.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARM = slice(0, 5)
CTRL_DROP_REF = 0.204  # ctrl 이 만든 낙폭 (0914 측정 🟢). 비교 기준점


def explained(
    state: list[np.ndarray], target: list[np.ndarray], horizon: float = 1.0
) -> tuple[float, float, float]:
    """Extrapolation explainability of a target given state history.
    state 이력만으로 타깃이 얼마나 설명되는지.

    ⚠️ 정정 2026-09-14 — `horizon` 이 없던 판(1스텝 고정)은 **k 스텝 앞 타깃을
    1스텝 외삽으로 맞히라고 시키는** 계측기였다. k 가 커질수록 기준선이 자동으로
    무너져 낙폭이 커지고, 그것이 "좋은 타깃"으로 오독된다. 고장난 계측기는 스스로
    알리지 않는다.

    horizon=k 면 예측은 `s[t] + k*(s[t] - s[t-1])` — "가던 대로 k 스텝 가면 어디".
    horizon=1 은 `2*s[t] - s[t-1]` 로 옛 정의와 같아, `probe_command_vs_action.py`
    의 수치와 그대로 대조된다."""
    ids, exs = [], []
    for s, a in zip(state, target):
        if s.shape[0] < 4:
            continue
        cur = s[1:-1]
        tgt = a[1:-1]
        pred = s[1:-1] + horizon * (s[1:-1] - s[:-2])
        ids.append(np.abs(tgt - cur).mean(axis=0))
        exs.append(np.abs(tgt - pred).mean(axis=0))
    if not ids:
        raise ValueError("에피소드가 전부 4틱 미만이다")
    i_m = np.median(np.stack(ids), axis=0)
    e_m = np.median(np.stack(exs), axis=0)
    expl = 1.0 - e_m[ARM].sum() / max(i_m[ARM].sum(), 1e-12)
    return float(expl), float(i_m[ARM].mean()), float(e_m[ARM].mean())


def lead_target(state: np.ndarray, k: int) -> np.ndarray:
    """state[min(t+k, T-1)]. make_lead_targets.py 와 같은 규약이어야 한다."""
    n = state.shape[0]
    return state[np.minimum(np.arange(n) + k, n - 1)]


def main() -> int:
    """Entry point.
    진입점."""
    ap = argparse.ArgumentParser(description="리드-k 타깃 설명력 스윕")
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--command", action="store_true", help="ctrl 사이드카도 함께 잰다")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.data)
    npzs = sorted(root.glob("*.npz"))
    if not npzs:
        print(f"!! .npz 없음: {root}")
        return 1

    states, actions, commands, miss = [], [], [], 0
    for f in npzs:
        with np.load(f, allow_pickle=True) as z:
            s = z["state"].astype(np.float64)
            a = z["action"].astype(np.float64)
        states.append(s)
        actions.append(a)
        if args.command:
            c = f.with_suffix(".command.npy")
            if not c.exists():
                miss += 1
                commands.append(None)
                continue
            cmd = np.load(c).astype(np.float64)
            if cmd.shape != s.shape:
                print(f"!! shape 불일치 {f.name}: state{s.shape} command{cmd.shape}")
                return 1
            commands.append(cmd)

    # 계측기 검증 — k=1 은 계약 action 과 (마지막 틱 제외) 같아야 한다
    t1 = [lead_target(s, 1) for s in states]
    dev = max(float(np.abs(t[:-1] - a[:-1]).max()) for t, a in zip(t1, actions))
    print(f"\n[ 계측기 검증 ] k=1 과 계약 action 의 최대 차이: {dev:.2e}", end="")
    if dev > 1e-4:
        print("  -> !! 실패. 계약 규약이 state[t+1] 이 아니다")
        return 1
    print("  -> 통과")

    print(f"\n에피소드 {len(states)}편 · 단위 계약 [-1,1] · 팔 5축")
    print("\n  설명력(1스텝) — 기준선이 1스텝 외삽 고정. k 가 크면 기준선이 무너진다")
    print("  설명력(k스텝) — 기준선이 s[t] + k*(s[t]-s[t-1]). **공정 비교는 이쪽이다**")
    print(f"\n  {'타깃':<20}{'크기':>10}{'설명력(1스텝)':>15}{'설명력(k스텝)':>15}{'리드/이동':>11}")

    base_expl = None
    rows = {}
    move_ref = float(np.median(
        [np.abs(a[1:-1] - s[1:-1]).mean() for s, a in zip(states, actions)]
    ))
    for k in args.k:
        tgt = [lead_target(s, k) for s in states]
        e1, i, x1 = explained(states, tgt, horizon=1.0)
        ek, _, xk = explained(states, tgt, horizon=float(k))
        lead = float(np.median([np.abs(t[1:-1] - s[1:-1]).mean() for s, t in zip(states, tgt)]))
        if k == 1:
            base_expl = ek
        rows[f"lead{k}"] = {"explained_1step": e1, "explained_kstep": ek,
                            "mag": i, "extrap_1step": x1, "extrap_kstep": xk,
                            "horizon": float(k),
                            "lead_ratio": lead / max(move_ref, 1e-12)}
        print(f"  {'state[t+' + str(k) + ']':<20}{i:>10.5f}{e1:>14.1%}{ek:>14.1%}"
              f"{lead / max(move_ref, 1e-12):>10.2f}x")

    if args.command and any(c is not None for c in commands):
        pair = [(s, c) for s, c in zip(states, commands) if c is not None]
        ss = [s for s, _ in pair]
        cc = [c for _, c in pair]
        lead = float(np.median([np.abs(c[1:-1] - s[1:-1]).mean() for s, c in pair]))
        ratio = lead / max(move_ref, 1e-12)
        # ctrl 은 정수 스텝이 아니다. 실측 리드 비율을 그대로 지평으로 쓴다
        e1, i, x1 = explained(ss, cc, horizon=1.0)
        ek, _, xk = explained(ss, cc, horizon=ratio)
        rows["command"] = {"explained_1step": e1, "explained_kstep": ek,
                           "mag": i, "extrap_1step": x1, "extrap_kstep": xk,
                           "horizon": ratio, "lead_ratio": ratio}
        print(f"  {'command = ctrl':<20}{i:>10.5f}{e1:>14.1%}{ek:>14.1%}{ratio:>10.2f}x")
        if miss:
            print(f"  (ctrl 사이드카 누락 {miss}편)")

    if base_expl is None:
        print("\n  !! k=1 이 목록에 없어 낙폭을 낼 수 없다")
        return 1

    cmd_row = rows.get("command")
    cmd_drop = base_expl - cmd_row["explained_kstep"] if cmd_row else CTRL_DROP_REF
    print(f"\n[ 낙폭 — k스텝 기준선. k=1 기준 {base_expl:.1%}, "
          f"ctrl 낙폭 {cmd_drop * 100:.1f}%p ]")
    for name, r in rows.items():
        if name == "lead1":
            continue
        d = base_expl - r["explained_kstep"]
        mark = "ctrl 이상" if d >= cmd_drop else "ctrl 미만"
        print(f"  {name:<12}{d * 100:>7.1f}%p   {mark}")

    print("\n  이 도구가 답하지 않는 것")
    print("     - 설명력이 낮다고 학습이 되는 것은 아니다. 롤아웃으로만 안다")
    print("     - 리드가 너무 크면 타깃이 관측과 무관해진다. 그것도 설명력은 낮게 나온다")
    print("     - 정책은 state 이력이 아니라 현재 state 한 장만 본다. 외삽 기준선은")
    print("       정책이 실제로 쓸 수 있는 것보다 강하다 — 설명력은 상한이다")
    print("     - 전부 시뮬이다. 실데이터에서 같은 k 가 최적이라는 보장은 없다\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "data": str(root), "n": len(states), "command_missing": miss,
            "move_ref": move_ref, "base_explained": base_expl,
            "ctrl_drop_ref": CTRL_DROP_REF, "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"결과: {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
