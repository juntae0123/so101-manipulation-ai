#!/usr/bin/env python3
"""Write lead-k target sidecars beside contract episodes. Never touches the .npz.
계약 에피소드 옆에 리드-k 타깃 사이드카를 쓴다. .npz 는 건드리지 않는다.

계약 0.3.0 은 `action[t] = state[t+1]` 이다 — **이미 간 거리**. 30Hz 에서 그 값은
관측 없이 87~94% 외삽되고, 롤아웃에서 정지가 고정점이 된다 (0914 확정 🟢).

`ctrl` 은 살아났지만 (0/300 -> 22/300) **실수집 UMI 에 대응물이 있는지 모른다**.
반면 `state[t+k]` 는 기존 계약 데이터만으로 계산된다 — 재수집도, 계약 변경도,
상대 트랙 답변도 필요 없다.

이 도구는 그 타깃을 `ep_XXXXX.lead{k}.npy` 로 쓴다. `read_episode` 도 `validate()` 도
이 파일을 보지 않으므로 **계약은 그대로다**. 학습은
`train_bc.py --target-sidecar lead{k}` 로 집어 쓴다.

종단 규약: `t + k` 가 에피소드를 넘으면 **마지막 state 로 클램프**한다. 에피소드를
자르지 않으므로 샘플 수가 k 에 따라 변하지 않는다 — 조건 간 비교가 성립하려면
샘플 수가 같아야 한다.

    # [서버]
    python tools/make_lead_targets.py --data datasets/sim_pick_cmd --k 2 4 8 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def lead_target(state: np.ndarray, k: int) -> np.ndarray:
    """state[min(t+k, T-1)] for every t. Clamped at the tail, never truncated.
    모든 t 에 대해 state[min(t+k, T-1)]. 끝은 클램프하고 자르지 않는다."""
    if state.ndim != 2:
        raise ValueError(f"state 는 (T, D) 여야 한다: {state.shape}")
    n = state.shape[0]
    idx = np.minimum(np.arange(n) + k, n - 1)
    return state[idx].astype(np.float32, copy=True)


def self_test() -> None:
    """Fixture. A broken instrument does not announce itself.
    픽스처. 고장난 계측기는 스스로 알리지 않는다."""
    s = np.arange(24, dtype=np.float32).reshape(4, 6)
    t1 = lead_target(s, 1)
    assert t1.shape == s.shape, t1.shape
    assert np.array_equal(t1[:3], s[1:]), "k=1 은 계약 action 과 같아야 한다"
    assert np.array_equal(t1[3], s[3]), "마지막은 클램프"
    t8 = lead_target(s, 8)
    assert np.array_equal(t8, np.repeat(s[-1:], 4, axis=0)), "k>T 는 전부 마지막"
    assert np.array_equal(s, np.arange(24, dtype=np.float32).reshape(4, 6)), "원본 불변"
    print("fixture: PASS")


def main() -> int:
    """Entry point.
    진입점."""
    ap = argparse.ArgumentParser(description="리드-k 타깃 사이드카 생성 (계약 불변)")
    ap.add_argument("--data", required=True, help="계약 에피소드 디렉터리")
    ap.add_argument("--k", type=int, nargs="+", required=True, help="리드 스텝 목록")
    ap.add_argument("--overwrite", action="store_true", help="기존 사이드카를 덮어쓴다")
    args = ap.parse_args()

    self_test()

    bad = [k for k in args.k if k < 1]
    if bad:
        print(f"!! k 는 1 이상이어야 한다: {bad}")
        return 1

    root = Path(args.data)
    npzs = sorted(root.glob("*.npz"))
    if not npzs:
        print(f"!! .npz 없음: {root}")
        return 1

    written = {k: 0 for k in args.k}
    skipped = {k: 0 for k in args.k}
    for f in npzs:
        with np.load(f, allow_pickle=True) as z:
            state = z["state"].astype(np.float32)
            action = z["action"].astype(np.float32)
        for k in args.k:
            dst = f.with_suffix(f".lead{k}.npy")
            if dst.exists() and not args.overwrite:
                skipped[k] += 1
                continue
            np.save(dst, lead_target(state, k))
            written[k] += 1
        # 계측기 검증: k=1 사이드카는 계약 action 과 마지막 틱만 달라야 한다
        if 1 in args.k:
            t1 = lead_target(state, 1)
            if not np.allclose(t1[:-1], action[:-1], atol=1e-6):
                print(f"!! k=1 이 계약 action 과 다르다: {f.name}")
                return 1

    print(f"\n에피소드 {len(npzs)}편 · {root}")
    for k in args.k:
        print(f"  lead{k:<3} 생성 {written[k]:>4}  건너뜀 {skipped[k]:>4}")

    meta = root / "LEAD_TARGETS.json"
    meta.write_text(json.dumps({
        "note": "실험용 사이드카. 계약 npz 는 건드리지 않았다",
        "targets": {f"lead{k}": "state[min(t+k, T-1)]" for k in args.k},
        "terminal_rule": "clamp_to_last_state",
        "episodes": len(npzs),
        "issue": "S15P21A103-170",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n메타: {meta}")
    print("  학습: train_bc.py --target-sidecar lead8\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
