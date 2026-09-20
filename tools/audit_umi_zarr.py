#!/usr/bin/env python3
"""Audit an official-UMI ReplayBuffer zarr without training anything.
공식 UMI zarr 를 학습 없이 잰다 — 편 수·길이·파지 개구·작업 반경·결측.

왜 필요한가
-----------
현석 배치(v4/review_v1)와 우리 v10 은 표현이 다르다. v10 은 상대 청크라 절대 pose 가
없고, 이쪽은 절대 EEF pose 가 있다. **같은 자로 재야 비교가 된다.**

측정 원칙
---------
- 모수를 같이 찍는다 (찾음 N / 전체 M)
- '없음' 과 '괜찮음' 이 같은 출력으로 나오지 않게 한다. 못 잰 건 못 쟀다고 찍는다
- 파지 개구는 **닫힘이 실제로 일어난 편만** 센다. 평평한 gap 은 거부한다

Usage
-----
  python audit_umi_zarr.py --selftest
  python audit_umi_zarr.py --zarr <경로> --out out/audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CLOSURE_MIN_DROP_M = 0.010      # 이보다 덜 움직인 gap 은 '닫힘 없음' 으로 거부
GAP_TOL_M = 0.002               # 최소값 근방 허용폭
KEYS = ("robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width")


class ClosureNotFound(ValueError):
    """No detectable gripper closure. 닫힘이 관측되지 않았다."""


def ci95(k: int, n: int) -> tuple[float, float]:
    """Wilson interval, percent. 윌슨 구간[%]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p, z = k / n, 1.959964
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def closure_gap_m(gap: np.ndarray, tol: float = GAP_TOL_M,
                  min_drop: float = CLOSURE_MIN_DROP_M) -> tuple[float, int]:
    """Gripper width at the moment of closure. 닫히는 순간의 개구[m] 와 그 행.

    닫힘이 없으면 예외다. 평평한 신호에 argmin 을 쓰면 0행이 조용히 나온다.
    """
    g = np.asarray(gap, dtype=np.float64).ravel()
    if g.size < 3:
        raise ClosureNotFound(f"길이 {g.size} — 너무 짧다")
    drop = float(g.max() - g.min())
    if drop < min_drop:
        raise ClosureNotFound(f"gap 변화 {drop * 1000:.1f}mm < {min_drop * 1000:.0f}mm — 닫힘 없음")
    c = int(np.argmax(g <= float(g.min()) + tol))
    if c == 0 or c == g.size - 1:
        raise ClosureNotFound(f"닫힘 행 {c} 가 경계다 (길이 {g.size})")
    return float(g[c]), c


def episode_bounds(ends: np.ndarray) -> list[tuple[int, int]]:
    """episode_ends -> [(start, end)). 편 경계."""
    e = np.asarray(ends, dtype=np.int64).ravel()
    starts = np.concatenate([[0], e[:-1]])
    return [(int(a), int(b)) for a, b in zip(starts, e) if b > a]


def audit(data: dict, ends: np.ndarray) -> dict:
    """Measure one dataset. 한 데이터셋을 잰다. 모수를 반드시 같이 낸다."""
    bounds = episode_bounds(ends)
    n_ep = len(bounds)
    pos = np.asarray(data["robot0_eef_pos"], dtype=np.float64)
    rot = np.asarray(data["robot0_eef_rot_axis_angle"], dtype=np.float64)
    gap = np.asarray(data["robot0_gripper_width"], dtype=np.float64).reshape(len(pos), -1)[:, 0]

    lens = [b - a for a, b in bounds]
    closures, rejected = [], []
    paths, spans = [], []
    for a, b in bounds:
        try:
            g, _ = closure_gap_m(gap[a:b])
            closures.append(g * 1000.0)
        except ClosureNotFound as exc:
            rejected.append(str(exc))
        p = pos[a:b]
        paths.append(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()))
        spans.append(float(np.linalg.norm(p.max(0) - p.min(0))))

    nan_rows = int(np.sum(~np.isfinite(np.c_[pos, rot, gap[:, None]]).all(axis=1)))
    ang = np.linalg.norm(rot, axis=1)

    out = {
        "frames": int(len(pos)),
        "episodes": n_ep,
        "episode_len": {"min": int(min(lens)), "median": float(np.median(lens)),
                        "max": int(max(lens))} if lens else None,
        "closure": {
            "scored": len(closures), "total": n_ep,
            "rejected": len(rejected),
            "reject_examples": sorted(set(rejected))[:3],
            "median_mm": float(np.median(closures)) if closures else None,
            "p10_mm": float(np.percentile(closures, 10)) if closures else None,
            "p90_mm": float(np.percentile(closures, 90)) if closures else None,
            "scored_rate_ci95": list(ci95(len(closures), n_ep)),
        },
        "eef_pos": {
            "min_m": pos.min(0).round(4).tolist(), "max_m": pos.max(0).round(4).tolist(),
            "extent_m": (pos.max(0) - pos.min(0)).round(4).tolist(),
            "per_episode_span_median_m": float(np.median(spans)) if spans else None,
            "per_episode_path_median_m": float(np.median(paths)) if paths else None,
        },
        "rot_axis_angle": {"norm_min_rad": float(ang.min()), "norm_max_rad": float(ang.max()),
                           "over_pi": int(np.sum(ang > np.pi + 1e-6))},
        "gripper_width": {"min_mm": float(gap.min()) * 1000, "max_mm": float(gap.max()) * 1000},
        "nonfinite_rows": nan_rows,
    }
    return out


def _synth(n_ep: int = 5, ln: int = 40, closing: bool = True) -> tuple[dict, np.ndarray]:
    """Known-answer dataset. 정답을 아는 합성 데이터 — 닫힘 개구 정확히 40.0mm."""
    pos, rot, gap = [], [], []
    for _ in range(n_ep):
        t = np.linspace(0, 1, ln)
        pos.append(np.c_[0.30 + 0.05 * t, 0.00 + 0.02 * t, 0.10 - 0.05 * t])
        rot.append(np.tile([0.0, 0.0, 1.0], (ln, 1)))
        g = np.full(ln, 0.085)
        if closing:
            g[ln // 2:] = 0.040           # 45mm 떨어진다 -> 닫힘으로 인정, 개구 40.0mm
        gap.append(g)
    ends = np.cumsum([ln] * n_ep)
    return ({"robot0_eef_pos": np.concatenate(pos),
             "robot0_eef_rot_axis_angle": np.concatenate(rot),
             "robot0_gripper_width": np.concatenate(gap)[:, None]}, ends)


def selftest() -> int:
    """Known-answer and discriminating rows. 정답 아는 행과 판별행."""
    log, bad = [], 0

    def chk(name: str, cond: bool, note: str = "") -> None:
        nonlocal bad
        log.append((name, bool(cond), note))
        if not cond:
            bad += 1

    d, e = _synth()
    r = audit(d, e)
    chk("1 편 수·프레임 모수", r["episodes"] == 5 and r["frames"] == 200,
        f"편 {r['episodes']} · 프레임 {r['frames']}")
    chk("2 닫힘 개구 정답 40.0mm", abs(r["closure"]["median_mm"] - 40.0) < 1e-6,
        f"{r['closure']['median_mm']:.4f} mm")
    chk("3 채점 모수 5/5", r["closure"]["scored"] == 5 and r["closure"]["total"] == 5,
        f"{r['closure']['scored']}/{r['closure']['total']}")

    d2, e2 = _synth(closing=False)
    r2 = audit(d2, e2)
    chk("4 닫힘 없는 데이터 -> 전부 거부 (판별행)",
        r2["closure"]["scored"] == 0 and r2["closure"]["median_mm"] is None,
        f"채점 {r2['closure']['scored']}/{r2['closure']['total']} · "
        f"{(r2['closure']['reject_examples'] or ['-'])[0]}")

    d3, e3 = _synth()
    d3["robot0_gripper_width"] = d3["robot0_gripper_width"].copy()
    d3["robot0_gripper_width"][7] = np.nan
    r3 = audit(d3, e3)
    chk("5 결측 행이 세어진다 (판별행)", r3["nonfinite_rows"] == 1, f"{r3['nonfinite_rows']}건")

    chk("6 경계 닫힘 거부", _boundary_rejected(), "첫 행부터 닫혀 있으면 거부해야 한다")

    ext = r["eef_pos"]["extent_m"]
    chk("7 작업 범위 정답 (0.05,0.02,0.05)",
        np.allclose(ext, [0.05, 0.02, 0.05], atol=1e-4), f"{ext}")

    for nm, ok, note in log:
        print(f"  {'OK ' if ok else 'FAIL'}  {nm}" + (f"   {note}" if note else ""))
    print(f"\n자체검증 {len(log) - bad}/{len(log)}")
    return 1 if bad else 0


def _boundary_rejected() -> bool:
    """Closure at row 0 must be refused. 0행 닫힘은 거부되어야 한다."""
    g = np.concatenate([[0.040], np.full(30, 0.085)])
    try:
        closure_gap_m(g)
        return False
    except ClosureNotFound:
        return True


def load_zarr(path: Path) -> tuple[dict, np.ndarray]:
    """Read arrays we need. 필요한 배열만 읽는다 (이미지는 건드리지 않는다)."""
    import zarr
    z = zarr.open(str(path), mode="r")
    missing = [k for k in KEYS if k not in z["data"]]
    if missing:
        raise SystemExit(f"!! 배열 누락 {missing} — 있는 것 {sorted(z['data'].array_keys())}")
    data = {k: np.asarray(z["data"][k]) for k in KEYS}
    ends = np.asarray(z["meta"]["episode_ends"])
    return data, ends


def main() -> None:
    """CLI entry point. 명령행 진입점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--zarr")
    ap.add_argument("--label", default=None, help="보고서에 적을 이름")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 수치를 내지 않는다")
    if not a.zarr:
        ap.error("--zarr 가 필요하다")

    p = Path(a.zarr).expanduser()
    data, ends = load_zarr(p)
    r = audit(data, ends)
    r["zarr"] = str(p)
    r["label"] = a.label or p.name

    c, ep = r["closure"], r["eef_pos"]
    print(f"\n[{r['label']}]  프레임 {r['frames']} · 편 {r['episodes']}")
    print(f"편 길이        최소 {r['episode_len']['min']} · 중앙 {r['episode_len']['median']:.0f} "
          f"· 최대 {r['episode_len']['max']}")
    print(f"파지 개구      채점 {c['scored']}/{c['total']}편 · 거부 {c['rejected']}편")
    if c["median_mm"] is not None:
        print(f"               중앙 {c['median_mm']:.1f} mm · p10 {c['p10_mm']:.1f} "
              f"· p90 {c['p90_mm']:.1f}  (폭 {c['p90_mm'] - c['p10_mm']:.1f} mm)")
    else:
        print("               ** 닫힘이 관측된 편이 0 이다. 개구를 내지 않는다 **")
    if c["reject_examples"]:
        for s in c["reject_examples"]:
            print(f"               거부 사유 예: {s}")
    print(f"EEF 범위[m]    x/y/z 폭 {ep['extent_m']}")
    print(f"               min {ep['min_m']}  max {ep['max_m']}")
    print(f"               편당 이동거리 중앙 {ep['per_episode_path_median_m']:.3f} m · "
          f"직선 span 중앙 {ep['per_episode_span_median_m']:.3f} m")
    print(f"그리퍼 폭      {r['gripper_width']['min_mm']:.1f} ~ {r['gripper_width']['max_mm']:.1f} mm")
    print(f"회전 크기      {r['rot_axis_angle']['norm_min_rad']:.3f} ~ "
          f"{r['rot_axis_angle']['norm_max_rad']:.3f} rad · pi 초과 "
          f"{r['rot_axis_angle']['over_pi']}건")
    print(f"결측 행        {r['nonfinite_rows']} / {r['frames']}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
