#!/usr/bin/env python3
"""Compare a trained BC checkpoint against observation-free baselines, in contract units.
학습된 BC 체크포인트를 관측 없는 기준선과 계약 단위로 비교한다.

왜: val loss 는 arm L1(표준화) + 그리퍼 BCE 가 섞인 단위라 기준선과 비교가 안 된다.
    BCPolicy.act() 는 계약 단위 [-1,1] 행동을 돌려주므로 환산이 필요 없다.

한계: 진짜 오픈루프 롤아웃이 아니다. 예측 state 에 해당하는 이미지가 없기 때문이다.
      여기서 재는 것은 1스텝 teacher-forced 오차다. 누적 오차는 못 잰다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.bc import BCPolicy          # noqa: E402
from sim.base import Observation        # noqa: E402

ARM = slice(0, 5)
GATE = 0.8   # BC 오차 <= 외삽 오차 * GATE 여야 "관측을 쓴다"


def build_obs(images: dict, state: np.ndarray, ts: float):
    """Construct an Observation from recorded arrays.
    기록된 배열로 Observation 을 만든다."""
    if not dataclasses.is_dataclass(Observation):
        raise TypeError("Observation 이 dataclass 가 아니다")
    kw = {}
    for f in dataclasses.fields(Observation):
        if "image" in f.name:
            kw[f.name] = images
        elif "time" in f.name:
            kw[f.name] = float(ts)
        elif "state" in f.name:
            kw[f.name] = state
    return Observation(**kw)


def val_indices(exp_log: Path, ckpt: str) -> list[int] | None:
    """Recover the val episode indices recorded for this checkpoint.
    이 체크포인트에 기록된 val 에피소드 인덱스를 되찾는다."""
    found = None
    if not exp_log.exists():
        return None
    for line in exp_log.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("experiment") != "train_bc":
            continue
        if str(d.get("result", {}).get("checkpoint", "")).endswith(ckpt):
            found = d.get("conditions", {}).get("val_episodes")
    return found


def main() -> int:
    """Entry point.
    진입점."""
    ap = argparse.ArgumentParser(description="BC ckpt vs 관측없는 기준선 (계약 단위)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True, nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.data)
    files = sorted(root.glob("*.npz")) or sorted(root.rglob("*.npz"))
    if not files:
        print(f"!! .npz 없음: {root}")
        return 1
    exp_log = Path("EXP_LOG.jsonl")

    results = {}
    for cp in args.ckpt:
        name = Path(cp).name
        try:
            pol = BCPolicy(Path(cp), device=args.device)
        except Exception as e:
            print(f"!! {name} 로드 실패: {type(e).__name__}: {e}")
            continue
        cams = list(pol.meta["camera_names"])
        idx = val_indices(exp_log, name)
        if idx is None:
            print(f"  ({name}: EXP_LOG 에 val_episodes 가 없다 → 전 에피소드로 잰다)")
            idx = list(range(len(files)))
        sel = [files[i] for i in idx if i < len(files)]

        bc_e, id_e, ex_e = [], [], []
        for f in sel:
            with np.load(f, allow_pickle=True) as z:
                s = z["state"].astype(np.float32)
                a = z["action"].astype(np.float32)
                imgs = {c: z[f"image__{c}"] for c in cams if f"image__{c}" in z.files}
                tss = z["state_timestamp"].astype(float) if "state_timestamp" in z.files else None
            if not imgs or s.shape[0] < 3:
                continue
            step = max(1, s.shape[0] // 25)
            for t in range(1, s.shape[0] - 1, step):
                obs = build_obs({c: imgs[c][t] for c in imgs}, s[t], tss[t] if tss is not None else t / 30.0)
                try:
                    pred = np.asarray(pol.act(obs), dtype=np.float32)
                except Exception as e:
                    print(f"!! act() 실패: {type(e).__name__}: {e}")
                    print("   Observation 필드:", [fl.name for fl in dataclasses.fields(Observation)])
                    return 2
                bc_e.append(float(np.abs(pred[ARM] - a[t][ARM]).mean()))
                id_e.append(float(np.abs(s[t][ARM] - a[t][ARM]).mean()))
                ex_e.append(float(np.abs((2 * s[t] - s[t - 1])[ARM] - a[t][ARM]).mean()))

        b, i, x = (float(np.mean(v)) for v in (bc_e, id_e, ex_e))
        clip = pol.n_clipped / max(pol.n_actions, 1)
        results[name] = {"bc": b, "identity": i, "extrap": x,
                         "n": len(bc_e), "n_val_ep": len(sel), "clip_rate": clip}
        print(f"\n[{name}]  val {len(sel)}편 · 표본 {len(bc_e)}")
        print(f"   BC        {b:.6f}")
        print(f"   identity  {i:.6f}")
        print(f"   외삽      {x:.6f}   <- 넘어야 할 선")
        print(f"   BC/외삽   {b/max(x,1e-12):.2f}x    포화율 {clip:.1%}")

    print(f"\n{'='*60}\n[ 판정 — 사전등록 게이트: BC <= 외삽 x {GATE} ]")
    for n, r in results.items():
        ok = r["bc"] <= r["extrap"] * GATE
        print(f"  {n:<28} {r['bc']/max(r['extrap'],1e-12):>6.2f}x  "
              f"{'통과 — 관측을 쓴다' if ok else '미달 — 외삽기다'}")
    print("\n  한계: 1스텝 teacher-forced 다. 누적 오차·접촉·파지는 여기 없다.\n")

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"결과: {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
