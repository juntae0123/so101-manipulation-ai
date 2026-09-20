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
    paths, spans, ratios = [], [], []
    for a, b in bounds:
        try:
            g, _ = closure_gap_m(gap[a:b])
            closures.append(g * 1000.0)
        except ClosureNotFound as exc:
            rejected.append(str(exc))
        p = pos[a:b]
        pl = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        sp = float(np.linalg.norm(p.max(0) - p.min(0)))
        paths.append(pl)
        spans.append(sp)
        ratios.append(pl / sp if sp > 1e-9 else float("inf"))

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
            "path_span_ratio_median": float(np.median(ratios)) if ratios else None,
        },
        "rot_axis_angle": {"norm_min_rad": float(ang.min()), "norm_max_rad": float(ang.max()),
                           "over_pi": int(np.sum(ang > np.pi + 1e-6))},
        "gripper_width": {"min_mm": float(gap.min()) * 1000, "max_mm": float(gap.max()) * 1000},
        "nonfinite_rows": nan_rows,
    }
    return out



# ── 게이트 (현장 수집 판정) ──────────────────────────────────────────────────
# ⚠️ 수집 **전에** 박은 값이다. 결과를 보고 옮기지 않는다.
#    참조 집단: 현석 s22_pick_v3 35편 (120 epoch 학습이 돌아간 배치) 와 v4 76편.
#    둘 다 통과하되 여유를 둔 선으로 잡았다. 못 맞추면 게이트를 낮추지 말고 다시 찍는다.
GATES = {
    "episodes_min":            {"v": 20,    "why": "1차 검증 표본. 단계식 수집 20 -> 검증 -> 100"},
    "nonfinite_rows_max":      {"v": 0,     "why": "현석 두 배치 모두 0/16328, 0/3475"},
    "closure_scored_rate_min": {"v": 1.0,   "why": "현석 76/76, 35/35. 닫힘이 안 잡히는 편은 학습에 못 쓴다"},
    "closure_spread_mm_max":   {"v": 3.0,   "why": "현석 1.4mm · v4 1.6mm. 2배 여유"},
    "episode_len_median_min":  {"v": 60,    "why": "30Hz 에서 2.0초. 너무 짧으면 접근이 안 담긴다"},
    "episode_len_median_max":  {"v": 150,   "why": "30Hz 에서 5.0초. 현석 99(3.3초), v4 216(7.2초)은 길다"},
    "path_span_ratio_max":     {"v": 2.5,   "why": "현석 1.63 · v4 2.90. 손을 덜 휘저어야 한다"},
    "eef_extent_m_max":        {"v": 0.30,  "why": "SO-101 도달 반경 0.3m. 축별 폭이 이보다 크면 못 따라간다"},
    "rot_span_rad_max":        {"v": 0.35,  "why": "현석 0.127 · v4 0.519. 접근 자세가 일정해야 5축으로 된다"},
    "rate_hz_range":           {"v": [29.5, 30.5], "why": "현석 30.0021Hz. 표본율이 흔들리면 다운샘플이 깨진다"},
}


def evaluate_gates(r: dict, rate_hz: float | None = None, gates: dict | None = None) -> dict:
    """Judge one audited dataset against pre-registered gates.
    감사 결과를 사전 등록 게이트로 판정한다. 모수를 같이 낸다."""
    g = {k: v["v"] for k, v in (gates or GATES).items()}
    c, ep = r["closure"], r["eef_pos"]
    rows = []

    def row(name: str, ok: bool | None, got, want) -> None:
        rows.append({"name": name, "ok": ok, "got": got, "want": want})

    row("편 수", r["episodes"] >= g["episodes_min"], r["episodes"], f">= {g['episodes_min']}")
    row("결측 행", r["nonfinite_rows"] <= g["nonfinite_rows_max"],
        f"{r['nonfinite_rows']} / {r['frames']}", f"<= {g['nonfinite_rows_max']}")
    rate = c["scored"] / c["total"] if c["total"] else 0.0
    row("파지 닫힘 채점률", rate >= g["closure_scored_rate_min"],
        f"{c['scored']} / {c['total']}", f">= {g['closure_scored_rate_min']:.0%}")
    if c["median_mm"] is None:
        row("파지 개구 폭", None, "닫힘 0편 — 대조 불가. 통과가 아니다",
            f"<= {g['closure_spread_mm_max']} mm")
    else:
        sp = c["p90_mm"] - c["p10_mm"]
        row("파지 개구 p10-p90 폭", sp <= g["closure_spread_mm_max"],
            f"{sp:.2f} mm (중앙 {c['median_mm']:.1f})", f"<= {g['closure_spread_mm_max']} mm")
    m = r["episode_len"]["median"] if r["episode_len"] else None
    row("편 길이 중앙", None if m is None else
        (g["episode_len_median_min"] <= m <= g["episode_len_median_max"]),
        f"{m} 프레임" if m is not None else "없음",
        f"{g['episode_len_median_min']}~{g['episode_len_median_max']}")
    ratio = ep["path_span_ratio_median"]
    row("경로/직선 비 중앙", None if ratio is None else ratio <= g["path_span_ratio_max"],
        f"{ratio:.2f}" if ratio is not None else "없음", f"<= {g['path_span_ratio_max']}")
    ext = max(ep["extent_m"])
    row("EEF 축별 최대 폭", ext <= g["eef_extent_m_max"], f"{ext:.3f} m",
        f"<= {g['eef_extent_m_max']} m")
    rs = r["rot_axis_angle"]["norm_max_rad"] - r["rot_axis_angle"]["norm_min_rad"]
    row("회전 크기 폭", rs <= g["rot_span_rad_max"], f"{rs:.3f} rad",
        f"<= {g['rot_span_rad_max']} rad")
    lo, hi = g["rate_hz_range"]
    if rate_hz is None:
        row("표본율", None, "미제공 — 대조 불가. 통과가 아니다 (--rate-hz)", f"{lo}~{hi} Hz")
    else:
        row("표본율", lo <= rate_hz <= hi, f"{rate_hz:.4f} Hz", f"{lo}~{hi} Hz")

    passed = sum(1 for x in rows if x["ok"] is True)
    failed = sum(1 for x in rows if x["ok"] is False)
    unknown = sum(1 for x in rows if x["ok"] is None)
    return {"rows": rows, "passed": passed, "failed": failed, "unknown": unknown,
            "total": len(rows),
            "verdict": "PASS" if failed == 0 and unknown == 0 else
                       ("FAIL" if failed else "INCOMPLETE")}


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

    # [8-9] 비교 모드 정답 아는 행. 같은 것은 같다고, 다른 것은 다르다고 해야 한다
    da, ea = _synth()
    db, eb = _synth()
    c_same = compare_arrays(da, db, ea, eb)
    chk("8 같은 데이터 -> 전부 동일", c_same["identical"] == c_same["total"],
        f"동일 {c_same['identical']}/{c_same['total']}")
    db2, eb2 = _synth()
    db2["robot0_eef_pos"] = db2["robot0_eef_pos"].copy()
    db2["robot0_eef_pos"][3, 0] += 0.001                 # 1mm 만 흔든다
    c_diff = compare_arrays(da, db2, ea, eb2)
    chk("9 1mm 차이를 잡아낸다 (판별행)",
        c_diff["identical"] == c_diff["total"] - 1,
        f"동일 {c_diff['identical']}/{c_diff['total']}")

    # [10-12] 게이트 정답 아는 행
    g_ok = evaluate_gates(audit(*_synth(n_ep=25, ln=100)), rate_hz=30.0)
    chk("10 좋은 데이터 -> PASS", g_ok["verdict"] == "PASS",
        f"통과 {g_ok['passed']}/{g_ok['total']} · {g_ok['verdict']}")
    g_few = evaluate_gates(audit(*_synth(n_ep=5, ln=100)), rate_hz=30.0)
    chk("11 편 5개 -> FAIL (판별행)", g_few["verdict"] == "FAIL",
        f"불합격 {g_few['failed']} · {g_few['verdict']}")
    g_nr = evaluate_gates(audit(*_synth(n_ep=25, ln=100)), rate_hz=None)
    chk("12 표본율 미제공 -> INCOMPLETE (판별행)", g_nr["verdict"] == "INCOMPLETE",
        "대조 불가가 통과로 나오면 안 된다")

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


def open_zarr(path: Path):
    """Open a zarr directory or .zip store. 디렉터리든 .zip 이든 연다."""
    import zarr
    p = str(path)
    if p.endswith(".zip"):
        return zarr.open(zarr.ZipStore(p, mode="r"), mode="r")
    return zarr.open(p, mode="r")


def load_zarr(path: Path) -> tuple[dict, np.ndarray]:
    """Read arrays we need. 필요한 배열만 읽는다 (이미지는 건드리지 않는다)."""
    z = open_zarr(path)
    have = sorted(z["data"].array_keys())
    missing = [k for k in KEYS if k not in have]
    if missing:
        raise SystemExit(f"!! 배열 누락 {missing} — 있는 것 {have}")
    data = {k: np.asarray(z["data"][k]) for k in KEYS}
    ends = np.asarray(z["meta"]["episode_ends"])
    return data, ends


def compare_arrays(a: dict, b: dict, ends_a: np.ndarray, ends_b: np.ndarray) -> dict:
    """Are two datasets the same data? 두 데이터셋이 같은 내용인가.

    zip 은 mtime 을 헤더에 담으므로 **같은 내용을 다시 압축만 해도 sha256 이 달라진다.**
    해시 차이는 내용 차이의 증거가 아니다. 배열을 직접 대조한다.
    """
    rows, same = [], 0
    pairs = [(k, a.get(k), b.get(k)) for k in KEYS] + [("episode_ends", ends_a, ends_b)]
    for k, x, y in pairs:
        if x is None or y is None:
            rows.append({"key": k, "verdict": "한쪽 없음", "equal": False}); continue
        x = np.asarray(x); y = np.asarray(y)
        if x.shape != y.shape:
            rows.append({"key": k, "verdict": f"형상 {x.shape} vs {y.shape}", "equal": False}); continue
        eq = bool(np.array_equal(x, y))
        d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
        same += eq
        rows.append({"key": k, "verdict": "동일" if eq else f"최대차 {d:.6g}",
                     "equal": eq, "max_abs_diff": d})
    return {"rows": rows, "identical": same, "total": len(pairs)}


def compare_images(pa: Path, pb: Path, n: int = 50) -> dict:
    """Sample-compare the image array. 이미지 배열을 표본으로 대조한다 (모수 병기)."""
    za, zb = open_zarr(pa), open_zarr(pb)
    if "camera0_rgb" not in za["data"] or "camera0_rgb" not in zb["data"]:
        return {"scored": 0, "total": 0, "note": "camera0_rgb 없음 — 대조 불가. 통과가 아니다"}
    A, B = za["data"]["camera0_rgb"], zb["data"]["camera0_rgb"]
    if A.shape != B.shape:
        return {"scored": 0, "total": 0, "note": f"형상 {A.shape} vs {B.shape} — 다르다"}
    total = A.shape[0]
    idx = np.unique(np.linspace(0, total - 1, min(n, total)).astype(int))
    eq = sum(int(np.array_equal(np.asarray(A[i]), np.asarray(B[i]))) for i in idx)
    return {"scored": eq, "total": int(len(idx)), "frames_total": int(total),
            "note": "표본 프레임 중 완전 동일한 개수"}


def main() -> None:
    """CLI entry point. 명령행 진입점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--zarr")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="두 zarr 의 배열 내용을 직접 대조한다 (해시가 아니라 값으로)")
    ap.add_argument("--image-samples", type=int, default=50)
    ap.add_argument("--gate", action="store_true", help="사전 등록 게이트로 판정")
    ap.add_argument("--rate-hz", type=float, default=None,
                    help="원본 표본율. report json 의 native_sample_rate_hz")
    ap.add_argument("--label", default=None, help="보고서에 적을 이름")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 수치를 내지 않는다")
    if a.compare:
        pa, pb = (Path(x).expanduser() for x in a.compare)
        da, ea = load_zarr(pa)
        db, eb = load_zarr(pb)
        cmp = compare_arrays(da, db, ea, eb)
        print(f"\nA {pa}\nB {pb}\n")
        for r in cmp["rows"]:
            print(f"  {r['key']:28} {r['verdict']}")
        print(f"\n포즈·개구 배열 동일 {cmp['identical']} / 전체 {cmp['total']}")
        img = compare_images(pa, pb, a.image_samples)
        if img["total"]:
            print(f"이미지 표본 동일 {img['scored']} / 표본 {img['total']} "
                  f"(전체 {img['frames_total']} 프레임)")
        else:
            print(f"이미지 대조 불가 — {img['note']}")
        allsame = cmp["identical"] == cmp["total"] and img["total"] and img["scored"] == img["total"]
        print("\n판정: " + ("같은 내용이다. 해시 차이는 재압축 때문이다 — A/B 가 아니다"
                          if allsame else
                          "내용이 다르다. 실제로 서로 다른 데이터다"))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(
                {"a": str(pa), "b": str(pb), "arrays": cmp, "images": img,
                 "identical_overall": bool(allsame)}, indent=2, ensure_ascii=False),
                encoding="utf-8")
            print(f"→ {a.out}")
        return

    if not a.zarr:
        ap.error("--zarr 또는 --compare 가 필요하다")

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

    if a.gate:
        gr = evaluate_gates(r, a.rate_hz)
        print("\n── 사전 등록 게이트 ──────────────────────────────────")
        for x in gr["rows"]:
            mark = {True: "통과", False: "불합격", None: "미판정"}[x["ok"]]
            print(f"  [{mark:^4}] {x['name']:<20} {str(x['got']):<34} 기준 {x['want']}")
        print(f"\n  통과 {gr['passed']} · 불합격 {gr['failed']} · 미판정 {gr['unknown']} "
              f"/ 전체 {gr['total']}")
        print(f"  판정: {gr['verdict']}"
              + ("   ← 미판정이 있으면 통과가 아니다" if gr["unknown"] else ""))
        r["gates"] = gr

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
