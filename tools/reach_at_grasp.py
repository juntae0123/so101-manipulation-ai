#!/usr/bin/env python3
"""Ask whether a demonstration's GRASP MOMENT is within the arm's reach, not the whole path.
시연의 **파지 순간**이 팔 도달 범위 안인지 묻는다. 전체 궤적이 아니다.

왜 파지 순간인가
----------------
이 프로젝트의 기조는 **결과 모방이지 궤적 모방이 아니다.** 사람이 크게 휘저어도
로봇은 같은 물체를 같게 집으면 된다. 접근 경로는 로봇이 스스로 짠다.
2026-09-17 실증 — 같은 데이터에서 전체 궤적 기준 29.2% 였다. 파지 기준은 다른 값이다.
**전체 궤적으로 폐기율을 계산하면 조용히 틀린다.**

이 계측기가 증명하는 것과 못 하는 것
------------------------------------
증명한다   반경·높이 밖의 점은 **어떤 자세로도 못 닿는다** (위치만으로 충분조건)
못 한다    반경 안이라고 닿는다는 뜻이 아니다. 자세·관절한계·자기충돌 미검사.
           → **"불가"는 확정, "가능"은 미확정**이다. 미판정을 통과로 쓰지 마라

Usage
    python reach_at_grasp.py --selftest
    python reach_at_grasp.py --zarr out/atlas_v1/x.zarr.zip --label atlas_v1 --out out/atlas_v1/reach.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_umi_zarr import (ClosureNotFound, closure_gap_m,  # noqa: E402
                            episode_bounds, load_zarr)

# ── 도달 모형 (configs/ 참조값. 하드코딩 금지 원칙상 전부 CLI 로 뺀다) ──────────
# SO-101 총질량 0.632kg · 도달 반경 대략 0.3m (프로젝트 지침).
# z 범위는 측면 파지 포락선 실측(z 0.027~0.051 에서 x 0.34~0.46)보다 넉넉히 잡는다.
DEF = {"r_min": 0.10, "r_max": 0.30, "z_min": 0.00, "z_max": 0.35,
       "window_s": 0.5, "rate_hz": 30.0, "grid": 41, "half_span": 0.60}


def reach_mask(pos: np.ndarray, base: np.ndarray, p: dict) -> np.ndarray:
    """Position-only necessary condition. 위치만 보는 필요조건. True = 불가라고 못 박을 수 없음."""
    d = np.linalg.norm(pos[:, :2] - base[None, :2], axis=1)
    d3 = np.linalg.norm(pos - base[None, :], axis=1)
    del d
    return (d3 >= p["r_min"]) & (d3 <= p["r_max"]) & \
           (pos[:, 2] >= p["z_min"]) & (pos[:, 2] <= p["z_max"])


def best_base(segments: list[np.ndarray], p: dict) -> tuple[np.ndarray, int, int]:
    """Grid-search the base placement that admits the most episodes.
    가장 많은 편을 받아들이는 베이스 위치를 격자 탐색한다. 반환 (base, 통과 편, 전체 편)."""
    if not segments:
        return np.zeros(3), 0, 0
    allp = np.concatenate(segments)
    cx, cy = float(allp[:, 0].mean()), float(allp[:, 1].mean())
    g, h = int(p["grid"]), float(p["half_span"])
    xs = np.linspace(cx - h, cx + h, g)
    ys = np.linspace(cy - h, cy + h, g)
    best, bn = np.array([cx, cy, 0.0]), -1
    for bx in xs:
        for by in ys:
            b = np.array([bx, by, 0.0])
            n = sum(1 for s in segments if reach_mask(s, b, p).all())
            if n > bn:
                bn, best = n, b
    return best, bn, len(segments)


def analyse(data: dict, ends: np.ndarray, p: dict) -> dict:
    """Measure both ways and report the parameters with every count. 모수를 반드시 같이 낸다."""
    bounds = episode_bounds(ends)
    pos = np.asarray(data["robot0_eef_pos"], dtype=np.float64)
    gap = np.asarray(data["robot0_gripper_width"], dtype=np.float64).reshape(len(pos), -1)[:, 0]
    w = max(1, int(round(p["window_s"] * p["rate_hz"])))

    grasp, whole, rejected = [], [], []
    for a, b in bounds:
        whole.append(pos[a:b])
        try:
            _, c = closure_gap_m(gap[a:b])
        except ClosureNotFound as exc:
            rejected.append(str(exc))
            continue
        lo, hi = max(a, a + c - w), min(b, a + c + w + 1)
        grasp.append(pos[lo:hi])

    gb, gn, gt = best_base(grasp, p)
    wb, wn, wt = best_base(whole, p)
    out = {
        "params": dict(p), "window_frames": 2 * w + 1,
        "episodes_total": len(bounds),
        "grasp": {"scored": gt, "rejected": len(rejected),
                  "episodes_ok": gn, "base": [round(float(x), 4) for x in gb],
                  "point_frac": float(np.concatenate([reach_mask(s, gb, p) for s in grasp]).mean())
                  if grasp else None},
        "whole": {"scored": wt, "episodes_ok": wn,
                  "base": [round(float(x), 4) for x in wb],
                  "point_frac": float(np.concatenate([reach_mask(s, wb, p) for s in whole]).mean())
                  if whole else None},
        "rejected_reasons": rejected[:5],
    }
    return out


def _synth(n_ep=5, n=60, radius=0.20, base=(0.0, 0.0, 0.0), far_eps=(), tail_far=False):
    """Known-answer data: every point sits at EXACTLY `radius` from `base`, z above 0.
    정답을 아는 합성 데이터 — 모든 점이 base 에서 정확히 radius 거리에 있다."""
    segs, gaps = [], []
    rng = np.random.default_rng(0)
    b = np.asarray(base, dtype=np.float64)
    for i in range(n_ep):
        r = 0.90 if i in far_eps else radius
        u = rng.normal(size=(n, 3))
        u[:, 2] = np.abs(u[:, 2]) + 0.3            # 위쪽 반구로 몰아 z > 0 을 보장한다
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        seg = b[None, :] + r * u                   # ||seg - b|| == r 이 정확히 성립
        if tail_far:
            # 닫힘은 n//2=30 행, 파지 창은 [15,45] 다. 47행부터 밀어야 창 밖이 된다
            seg[47:, 0] += 1.5                     # 끝부분만 멀리 — 전체 궤적만 떨어져야 한다
        segs.append(seg)
        g = np.full(n, 0.070); g[n // 2:] = 0.040  # 중간에 닫힌다
        gaps.append(g)
    pos = np.concatenate(segs)
    ends = np.cumsum([len(x) for x in segs])
    return {"robot0_eef_pos": pos, "robot0_gripper_width": np.concatenate(gaps)[:, None]}, ends


def selftest() -> int:
    ok = tot = 0

    def chk(name, cond, note=""):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(f"  {'OK  ' if cond else '실패'} {name}   {note}")

    p = dict(DEF); p["grid"] = 13; p["half_span"] = 0.4

    d, e = _synth(radius=0.20)
    r = analyse(d, e, p)
    chk("1 반경 0.20 -> 전 편 도달", r["grasp"]["episodes_ok"] == 5,
        f"편 {r['grasp']['episodes_ok']}/{r['grasp']['scored']}")

    d, e = _synth(radius=0.90)
    r = analyse(d, e, p)
    chk("2 반경 0.90 -> 0편 (판별행)", r["grasp"]["episodes_ok"] == 0,
        f"편 {r['grasp']['episodes_ok']}/{r['grasp']['scored']}")

    d, e = _synth(radius=0.20, far_eps=(2,))
    r = analyse(d, e, p)
    chk("3 1편만 멀다 -> 4/5 (판별행)", r["grasp"]["episodes_ok"] == 4,
        f"편 {r['grasp']['episodes_ok']}/{r['grasp']['scored']}")

    d, e = _synth(radius=0.20, base=(0.5, -0.3, 0.0))
    r = analyse(d, e, p)
    b = r["grasp"]["base"]
    chk("4 베이스를 옮겨 찾아낸다 (정답 아는 행)",
        r["grasp"]["episodes_ok"] == 5 and abs(b[0] - 0.5) < 0.15 and abs(b[1] + 0.3) < 0.15,
        f"편 {r['grasp']['episodes_ok']}/5 · base {b[:2]}")

    d, e = _synth(radius=0.20, tail_far=True)
    r = analyse(d, e, p)
    chk("5 파지 기준 != 전체 궤적 기준 (판별행)",
        r["grasp"]["episodes_ok"] == 5 and r["whole"]["episodes_ok"] == 0,
        f"파지 {r['grasp']['episodes_ok']}/5 · 전체 {r['whole']['episodes_ok']}/5")

    d, e = _synth(radius=0.20)
    d["robot0_gripper_width"] = np.full((len(d["robot0_eef_pos"]), 1), 0.070)   # 닫힘 없음
    r = analyse(d, e, p)
    chk("6 닫힘 없음 -> 채점에서 빠지고 모수에 남는다 (판별행)",
        r["grasp"]["scored"] == 0 and r["grasp"]["rejected"] == 5 and r["episodes_total"] == 5,
        f"채점 {r['grasp']['scored']} · 거부 {r['grasp']['rejected']} / 전체 {r['episodes_total']}")

    chk("7 모수·조건 보고", "params" in r and r["window_frames"] > 0,
        f"창 {r['window_frames']}프레임 · r_max {r['params']['r_max']}")

    print(f"\n자체검증 {ok}/{tot}")
    return 0 if ok == tot else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--zarr")
    ap.add_argument("--label", default="dataset")
    ap.add_argument("--out")
    for k, v in DEF.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if not a.zarr:
        ap.error("--zarr 가 필요하다 (또는 --selftest)")

    print("계측기 자체검증 먼저 —")
    if selftest() != 0:
        raise SystemExit("!! 자체검증 실패 — 실데이터를 재지 않는다")

    p = {k: getattr(a, k) for k in DEF}
    data, ends = load_zarr(Path(a.zarr).expanduser())
    r = analyse(data, ends, p)

    print(f"\n[{a.label}]  편 {r['episodes_total']}")
    print(f"조건           r {p['r_min']}~{p['r_max']} m · z {p['z_min']}~{p['z_max']} m · "
          f"파지 창 ±{p['window_s']}s ({r['window_frames']}프레임) · 격자 {p['grid']}^2")
    for key, ko in (("grasp", "파지 순간"), ("whole", "전체 궤적")):
        d = r[key]
        pf = "산출 불가" if d["point_frac"] is None else f"{d['point_frac'] * 100:.1f}%"
        print(f"{ko:<10}   편 {d['episodes_ok']} / {d['scored']}"
              + (f" (거부 {d['rejected']})" if "rejected" in d else "")
              + f" · 점 {pf} · 최적 base {d['base'][:2]}")
    if r["grasp"]["rejected"]:
        print("  거부 사유 예:", r["rejected_reasons"][0])
    print("\n** 반경 밖은 어떤 자세로도 불가다. 반경 안은 **미확정**이다 "
          "— 자세·관절한계 미검사. 통과로 쓰지 마라 **")
    if r["episodes_total"] and r["grasp"]["scored"]:
        if r["grasp"]["episodes_ok"] == 0:
            print("** 파지 기준으로도 0편이다. 베이스 위치로는 해결 안 된다 **")
        elif r["whole"]["episodes_ok"] < r["grasp"]["episodes_ok"]:
            print("** 전체 궤적 기준이 더 낮다 — 그 차이를 폐기율로 쓰면 틀린다 **")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"label": a.label, "zarr": a.zarr, **r},
                                          indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
