"""Compare two official-UMI zarr outputs produced on different machines.
서로 다른 기계에서 만든 공식 UMI zarr 산출물 두 개를 대조한다.

왜 (2026-09-18 · D-AI-62)
-------------------------
트랙 A 가 기준 산출물을 만들고 트랙 B 가 재실행한 결과를 대조해, 이후 배치부터
트랙 B 가 `convert_v10_to_umi.py` 를 단독 실행해도 되는지 판정한다.

**"동일 코드" 는 동일 바이너리가 아니다.** 두 트랙이 다른 기계에서 돌린다.
numpy·scipy 버전, BLAS 구현(OpenBLAS/MKL), 스레드 수에 따라 마지막 비트가 달라진다.
그래서 불일치가 났을 때 **"규약이 틀렸다" 와 "빌드가 다르다" 를 구분할 수 있어야 한다.**

판정 3상태 (양 트랙 합의)
-------------------------
    SCHEMA_MISMATCH        키 · shape · dtype · 원소 수 불일치
    NUMERICALLY_DIFFERENT  스키마는 같고 내용이 다름
    BIT_IDENTICAL          정규화 후 바이트 완전 일치

`camera0_rgb` 는 합격 조건에서 뺀다
-----------------------------------
1920x1080 → 224x224 리사이즈는 구현마다 다르다 (OpenCV 버전 · 보간 경로 · SIMD).
JPEG 디코더 버전 차이도 더해진다. **정상 실행도 계속 불합격이 난다.**

    필수 검사   shape · dtype · 원소 수
    진단값만    최대 절대차 · 평균 절대차 · 변경 픽셀 비율 · SHA-256
    입력 동일성 **원본 이미지 해시 목록으로 따로 확인한다** (황도경 제안)

⚠️ rgb 를 비트 대조에서 빼면 "완전히 다른 영상인데 shape 만 같아서 통과" 가 생긴다.
   그 구멍은 원본 해시가 막는다. `--source-hashes` 를 주지 않으면 그 항목은
   **"판정 불가" 로 출력한다. 통과가 아니다.**

계측기 규칙
-----------
- 모수를 같이 찍는다. 편수 · 배열 키 수를 병기해 `0개 불일치` 와 `0개 비교` 를 구분
- 정답을 아는 행을 넣는다. 마지막 비트만 다른 배열, dtype 만 다른 배열 등
- 측정 후 합격 기준을 완화하지 않는다. 결과와 환경 차이를 그대로 기록한다

Usage
-----
  python check_zarr_parity.py --selftest
  python check_zarr_parity.py --ref 기준.zarr --new 재실행.zarr --out parity.json
  python check_zarr_parity.py --ref A --new B --source-hashes ref_src.json new_src.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

RGB_KEYS = ("camera0_rgb",)          # 합격 조건에서 제외. 진단값만 낸다


# ── 순수 계산 — 자체검증 대상 ─────────────────────────────────────────────

def normalize(a: np.ndarray) -> np.ndarray:
    """C-contiguous + little-endian. 바이트 정규화 (양 트랙 합의 규약)."""
    if a.dtype.byteorder == ">" or (a.dtype.byteorder == "=" and sys.byteorder == "big"):
        a = a.astype(a.dtype.newbyteorder("<"))
    return np.ascontiguousarray(a)


def array_sha256(a: np.ndarray) -> str:
    """SHA-256 of the normalized bytes. 정규화 후 바이트의 SHA-256."""
    return hashlib.sha256(normalize(a).tobytes()).hexdigest()


def has_nonfinite(a: np.ndarray) -> bool:
    """NaN or Inf present? float 배열에 NaN·Inf 가 있나."""
    return bool(np.issubdtype(a.dtype, np.floating) and not np.all(np.isfinite(a)))


def first_mismatch(a: np.ndarray, b: np.ndarray) -> tuple | None:
    """Index of the first differing element, with both values.
    최초로 다른 원소의 인덱스와 양쪽 값. 같으면 None."""
    if a.shape != b.shape:
        return None
    d = a != b
    if not d.any():
        return None
    idx = np.unravel_index(int(np.argmax(d)), a.shape)
    return (tuple(int(i) for i in idx), a[idx].item(), b[idx].item())


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute difference. 최대 절대차. 형상이 다르면 NaN."""
    if a.shape != b.shape:
        return float("nan")
    if a.dtype == bool or b.dtype == bool:
        return float(np.count_nonzero(a != b))
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0


def rgb_diagnostics(a: np.ndarray, b: np.ndarray) -> dict:
    """Diagnostic-only numbers for image arrays. 이미지 배열의 진단값 — 판정에 안 쓴다."""
    if a.shape != b.shape:
        return {"comparable": False}
    fa, fb = a.astype(np.float64), b.astype(np.float64)
    d = np.abs(fa - fb)
    return {"comparable": True,
            "max_abs_diff": float(d.max()) if d.size else 0.0,
            "mean_abs_diff": float(d.mean()) if d.size else 0.0,
            "changed_pixel_ratio": float(np.count_nonzero(d) / d.size) if d.size else 0.0}


def classify(ref: dict, new: dict) -> str:
    """Three-state verdict per array. 배열 하나에 대한 3상태 판정."""
    if (ref["shape"] != new["shape"] or ref["dtype"] != new["dtype"]
            or ref["size"] != new["size"]):
        return "SCHEMA_MISMATCH"
    return "BIT_IDENTICAL" if ref["sha256"] == new["sha256"] else "NUMERICALLY_DIFFERENT"


def env_info() -> dict:
    """Record both sides' build environment. 양쪽 빌드 환경을 기록한다.
    불일치를 규약 오류가 아니라 환경 차이로 귀속할 수 있어야 한다."""
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "byteorder": sys.byteorder,
            "numpy": np.__version__}
    for mod in ("scipy", "cv2", "zarr"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:                                          # noqa: BLE001
            info[mod] = "없음"
    try:
        cfg = np.show_config(mode="dicts")
        info["blas"] = str(cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "?"))
    except Exception:                                              # noqa: BLE001
        info["blas"] = "조회 실패"
    for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        info[v] = os.environ.get(v, "미설정")
    return info


# ── 자체검증 ─────────────────────────────────────────────────────────────

def _meta(a: np.ndarray) -> dict:
    return {"shape": list(a.shape), "dtype": str(a.dtype), "size": int(a.size),
            "sha256": array_sha256(a)}


def selftest() -> int:
    bad = 0
    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 8, 10)).astype(np.float32)

    v = classify(_meta(base), _meta(base.copy()))
    ok = v == "BIT_IDENTICAL"
    print(f"[1] 동일 배열 → {v}  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    tweak = base.copy()
    tweak[3, 2, 7] = np.nextafter(tweak[3, 2, 7], np.float32(1e30))
    v = classify(_meta(base), _meta(tweak))
    d = max_abs_diff(base, tweak); fm = first_mismatch(base, tweak)
    ok = v == "NUMERICALLY_DIFFERENT" and 0 < d < 1e-5 and fm is not None and fm[0] == (3, 2, 7)
    print(f"[2] 마지막 비트만 다름 → {v} · 최대절대차 {d:.3e} · 최초 index {fm[0] if fm else None}  ", end="")
    print("OK" if ok else "!! 실패 — 비트 차이를 못 잡거나 index 가 틀렸다"); bad += (not ok)

    v = classify(_meta(base), _meta(base.astype(np.float64)))
    ok = v == "SCHEMA_MISMATCH"
    print(f"[3] dtype 만 다름 → {v}  ", end="")
    print("OK" if ok else "!! 실패 — dtype 차이를 내용 차이로 본다"); bad += (not ok)

    v = classify(_meta(base), _meta(base.reshape(50, 10, 8)))
    ok = v == "SCHEMA_MISMATCH"
    print(f"[4] shape 만 다름(원소수 동일) → {v}  ", end="")
    print("OK" if ok else "!! 실패 — 같은 원소수라 통과시킨다"); bad += (not ok)

    be = base.astype(base.dtype.newbyteorder(">"))
    ok = array_sha256(base) == array_sha256(be)
    print(f"[5] 엔디안만 다른 같은 값 → 해시 {'일치' if ok else '불일치'}  ", end="")
    print("OK" if ok else "!! 실패 — 정규화가 동작하지 않는다"); bad += (not ok)

    ok = array_sha256(base) == array_sha256(np.ascontiguousarray(base))
    print(f"[6] C-contiguous 정규화  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    nanned = base.copy(); nanned[0, 0, 0] = np.nan
    ok = has_nonfinite(nanned) and not has_nonfinite(base)
    print(f"[7] NaN 검출 · 정상 배열은 통과  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    img = rng.integers(0, 255, size=(10, 224, 224, 3), dtype=np.uint8)
    img2 = img.copy(); img2[0, 0, 0, 0] = (int(img2[0, 0, 0, 0]) + 1) % 256
    dg = rgb_diagnostics(img, img2)
    ok = (dg["comparable"] and dg["max_abs_diff"] == 1.0
          and 0 < dg["changed_pixel_ratio"] < 1e-5)
    print(f"[8] rgb 진단값 → 최대절대차 {dg['max_abs_diff']} · 변경비율 {dg['changed_pixel_ratio']:.2e}  ", end="")
    print("OK" if ok else f"!! 실패 — {dg}"); bad += (not ok)

    # 정답을 아는 오답: 완전히 다른 영상인데 shape 은 같다 → rgb 판정만으로는 못 잡는다
    other = rng.integers(0, 255, size=(10, 224, 224, 3), dtype=np.uint8)
    dg = rgb_diagnostics(img, other)
    ok = dg["changed_pixel_ratio"] > 0.9
    print(f"[9] 완전히 다른 영상 → 변경비율 {dg['changed_pixel_ratio']:.3f} (>0.9)  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)
    print("     ⚠️ 이건 진단값일 뿐 불합격 사유가 아니다. 입력 동일성은 원본 해시로 막는다")

    e = env_info()
    ok = e["numpy"] != "" and "OPENBLAS_NUM_THREADS" in e
    print(f"[10] 환경 기록 → numpy {e['numpy']} · blas {e['blas']} · byteorder {e['byteorder']}  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


# ── 대조 ─────────────────────────────────────────────────────────────────

def read_arrays(root: Path) -> dict[str, np.ndarray]:
    """Load every array under a zarr root. zarr 루트의 모든 배열을 읽는다."""
    import zarr
    g = zarr.open(str(root), mode="r")
    out: dict[str, np.ndarray] = {}

    def walk(node, prefix=""):
        for k in node.array_keys():
            out[f"{prefix}{k}"] = np.asarray(node[k])
        for k in node.group_keys():
            walk(node[k], f"{prefix}{k}/")
    walk(g)
    return out


def compare_source_hashes(ref_p: Path | None, new_p: Path | None) -> dict:
    """Same input videos? 입력 영상이 같은가. 미제공이면 '판정 불가' — 통과가 아니다."""
    if ref_p is None or new_p is None:
        return {"verdict": "판정 불가", "note": "--source-hashes 미제공. 통과로 처리하지 않는다"}
    a = json.loads(ref_p.read_text(encoding="utf-8"))
    b = json.loads(new_p.read_text(encoding="utf-8"))
    ka, kb = set(a), set(b)
    same = sorted(k for k in ka & kb if a[k] == b[k])
    diff = sorted(k for k in ka & kb if a[k] != b[k])
    return {"verdict": "일치" if (ka == kb and not diff) else "불일치",
            "matched": len(same), "differing": len(diff),
            "only_ref": sorted(ka - kb)[:10], "only_new": sorted(kb - ka)[:10],
            "total_ref": len(ka), "total_new": len(kb),
            "differing_keys": diff[:10]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ref", type=Path, help="기준 산출물 zarr (트랙 A)")
    ap.add_argument("--new", type=Path, help="재실행 산출물 zarr (트랙 B)")
    ap.add_argument("--source-hashes", type=Path, nargs=2, default=None,
                    metavar=("REF_JSON", "NEW_JSON"),
                    help="원본 이미지 해시 목록 2개. 입력 동일성 확인용")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 판정하지 않는다")
    print()

    if not (a.ref and a.new):
        raise SystemExit("!! --ref 와 --new 가 필요하다")

    ref = read_arrays(a.ref)
    new = read_arrays(a.new)
    keys = sorted(set(ref) | set(new))
    print(f"배열 키 — 기준 {len(ref)} · 재실행 {len(new)} · 합집합 {len(keys)}")
    if not keys:
        raise SystemExit("!! 배열 0개. **없는 게 아니라 못 읽은 것일 수 있다** — 경로를 확인하라")

    sh = compare_source_hashes(*(a.source_hashes or (None, None)))
    print(f"\n입력 동일성 (원본 이미지 해시) — {sh['verdict']}"
          + (f"  일치 {sh['matched']}/{sh['total_ref']} · 불일치 {sh['differing']}"
             if "matched" in sh else f"  {sh['note']}"))

    rows, counts = [], {"BIT_IDENTICAL": 0, "NUMERICALLY_DIFFERENT": 0,
                        "SCHEMA_MISMATCH": 0, "MISSING": 0}
    nonfinite = []
    print()
    for k in keys:
        if k not in ref or k not in new:
            counts["MISSING"] += 1
            side = "재실행에 없음" if k in ref else "기준에 없음"
            rows.append({"key": k, "verdict": "SCHEMA_MISMATCH", "note": side})
            print(f"  {k:36s} SCHEMA_MISMATCH  ({side})")
            continue
        ra, na = ref[k], new[k]
        if has_nonfinite(ra) or has_nonfinite(na):
            nonfinite.append(k)
        mr, mn = _meta(ra), _meta(na)
        v = classify(mr, mn)
        is_rgb = k.split("/")[-1] in RGB_KEYS
        row = {"key": k, "verdict": v, "rgb_excluded_from_gate": is_rgb,
               "ref": mr, "new": mn}
        if v != "BIT_IDENTICAL" and mr["shape"] == mn["shape"]:
            row["max_abs_diff"] = max_abs_diff(ra, na)
            fm = first_mismatch(ra, na)
            if fm:
                row["first_mismatch"] = {"index": fm[0], "ref": fm[1], "new": fm[2]}
        if is_rgb:
            row["rgb_diag"] = rgb_diagnostics(ra, na)
        rows.append(row)
        # 집계는 실제 판정을 그대로 센다. 게이트 제외는 gate_fail 에서만 적용한다
        counts[v] += 1
        tag = "  [rgb·판정제외]" if is_rgb and v == "NUMERICALLY_DIFFERENT" else ""
        extra = ""
        if "max_abs_diff" in row:
            extra = f"  최대절대차 {row['max_abs_diff']:.6e}"
            if "first_mismatch" in row:
                extra += f" · 최초 {row['first_mismatch']['index']}"
        print(f"  {k:36s} {v}{extra}{tag}")

    gate_fail = [r for r in rows
                 if r["verdict"] == "SCHEMA_MISMATCH"
                 or (r["verdict"] == "NUMERICALLY_DIFFERENT"
                     and not r.get("rgb_excluded_from_gate"))]

    print(f"\n{'='*70}")
    print(f"판정 집계 — 비교 {len(rows)} / 합집합 {len(keys)}")
    for k, v in counts.items():
        print(f"  {k:24s} {v}")
    if nonfinite:
        print(f"  ⚠️ NaN·Inf 포함 배열 {len(nonfinite)}개: {nonfinite[:5]}")

    verdict = "합격" if not gate_fail and not nonfinite else "불합격"
    if sh["verdict"] == "판정 불가":
        verdict += " (단, 입력 동일성 미확인)"
    elif sh["verdict"] == "불일치":
        verdict = "불합격 — 입력 영상이 다르다"
    print(f"\n최종: **{verdict}**   게이트 불합격 배열 {len(gate_fail)} / 비교 {len(rows)}")
    print("⚠️ 측정 후 합격 기준을 완화하지 않는다. 불일치는 그대로 기록하고 환경 차이로 귀속한다")

    if a.out:
        a.out.write_text(json.dumps(
            {"ref": str(a.ref), "new": str(a.new), "verdict": verdict,
             "source_hashes": sh, "counts": counts, "nonfinite": nonfinite,
             "gate_failures": len(gate_fail), "compared": len(rows),
             "keys_union": len(keys), "env_this_side": env_info(), "rows": rows},
            indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
