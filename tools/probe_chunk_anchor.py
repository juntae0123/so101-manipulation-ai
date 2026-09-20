#!/usr/bin/env python3
"""Is action[i,k] anchored at chunk start, or accumulated within the chunk?
청크 안의 액션이 '청크 시작 기준'인가 '누적'인가를 가른다.

왜 필요한가
-----------
2026-09-19 실물에서 팔이 물체로 전진하지 않고 반경 565mm 원호만 돌았다.
IK 잔차는 0.0007~0.029mm 였다 — **목표가 틀렸지 IK 가 틀린 게 아니다.**
로그에서 approach_error 가 청크 안에서 단조 증가했다:

    cycle 0   0.11 -> 0.80 -> 0.61 -> 0.53
    cycle 1   0.55 -> 1.16 -> 2.22 -> 3.57
    cycle 2   0.89 -> 1.57 -> 2.78 -> 4.44

재관측하면 리셋되고 청크 안에서 다시 커진다. **누적 오차의 모양이다.**

두 해석이 있다.
    누적(A)  T <- T @ A[i,k]        청크 안에서 앞 점 위에 또 곱한다
    앵커(B)  P_k = T_i @ A[i,k]     k 가 몇이든 전부 청크 시작 기준

변환기(convert_v10_to_umi.chain_consistency)는 **앵커**로 검증한다:
    pred = chain[i] @ row_to_transform(action[i, k])   ==   chain[i+k+1]

그런데 so101_infer.unroll 은 **누적**한다:
    for row in ch[lo:hi]:  T = T @ A

둘 다 맞을 수는 없다. 이 도구가 실 데이터로 가른다. 실물·MuJoCo 를 쓰지 않는다.

판정
----
    B 만 0 에 가깝다   -> 앵커가 맞다. so101_infer.unroll 이 틀렸다. 코드 수정으로 끝난다
    A 만 0 에 가깝다   -> 누적이 맞다. 변환기의 교차검증이 틀렸다
    둘 다 크다        -> 제3의 규약이다. 어느 쪽도 결론으로 쓰지 마라
    둘 다 0           -> 데이터가 두 해석을 구분하지 못한다 (델타가 너무 작다)

Usage
-----
  python probe_chunk_anchor.py --selftest
  python probe_chunk_anchor.py --dataset <v10 디렉터리> --limit 20 --out out/anchor.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

EXEC_LO, EXEC_HI = 1, 5          # 배포 계약 execSlice [1,5)


def rot6d_to_matrix(a6: np.ndarray) -> np.ndarray:
    """6D rotation -> 3x3, rows convention. r0,r1 은 회전행렬의 **행**이다."""
    a = np.asarray(a6, dtype=np.float64).reshape(2, 3)
    n0 = np.linalg.norm(a[0])
    if n0 < 1e-12:
        raise ValueError(f"r0 크기가 0에 가깝다: {n0:.3e}")
    b0 = a[0] / n0
    b1 = a[1] - float(b0 @ a[1]) * b0
    n1 = np.linalg.norm(b1)
    if n1 < 1e-12:
        raise ValueError(f"r0 와 r1 이 평행하다: {n1:.3e}")
    b1 = b1 / n1
    return np.column_stack([b0, b1, np.cross(b0, b1)]).T


def row_to_transform(row: np.ndarray) -> np.ndarray:
    """One action row -> 4x4. 액션 한 행을 동차변환으로."""
    r = np.asarray(row, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = rot6d_to_matrix(r[3:9])
    T[:3, 3] = r[:3]
    return T


def geodesic_deg(A: np.ndarray, B: np.ndarray) -> float:
    """Angle between two rotations, degrees. 두 회전 사이 측지각[도]."""
    c = (float(np.trace(A.T @ B)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def reconstruct_chain(action: np.ndarray) -> list[np.ndarray]:
    """Absolute chain from k=0 steps. k=0 스텝만으로 절대 연쇄를 만든다."""
    T = np.eye(4)
    chain = [T.copy()]
    for i in range(action.shape[0]):
        T = T @ row_to_transform(action[i, 0])
        chain.append(T.copy())
    return chain


SIDES = ("accum", "anchor", "deployed")


def probe_episode(action: np.ndarray, lo: int = EXEC_LO, hi: int = EXEC_HI) -> dict:
    """Score three formulas against the reconstructed chain.
    세 공식을 복원 연쇄와 대조한다. 모수(비교 건수)를 같이 낸다.

    accum     T0 @ A[0] @ A[1] ... @ A[k]   누적 규약의 정의. k=0 부터 전부 곱한다
    anchor    T0 @ A[k]                     전부 청크 시작 기준
    deployed  T0 @ A[lo] ... @ A[k]         so101_infer.unroll 이 **실제로** 하는 것
                                            (ch[lo:hi] 를 곱하므로 k<lo 를 버린다)
    """
    chain = reconstruct_chain(action)
    S = {n: {"p": [], "r": []} for n in SIDES}
    by_k: dict[int, dict] = {}
    for i in range(action.shape[0]):
        if i + hi >= len(chain):
            break
        T0 = chain[i]
        T_acc = T0.copy()
        T_dep = T0.copy()
        for k in range(0, min(hi, action.shape[1])):
            A = row_to_transform(action[i, k])
            T_acc = T_acc @ A
            if k >= lo:
                T_dep = T_dep @ A
            else:
                continue
            j = i + k + 1
            if j >= len(chain):
                break
            ref = chain[j]
            cand = {"accum": T_acc, "anchor": T0 @ A, "deployed": T_dep}
            d = by_k.setdefault(k, {n: [] for n in SIDES})
            for n, M in cand.items():
                pos = float(np.linalg.norm(M[:3, 3] - ref[:3, 3])) * 1000.0
                S[n]["p"].append(pos)
                S[n]["r"].append(geodesic_deg(M[:3, :3], ref[:3, :3]))
                d[n].append(pos)
    if not S["accum"]["p"]:
        return {"comparisons": 0}
    out = {"comparisons": len(S["accum"]["p"])}
    for n, v in S.items():
        out[n] = {"pos_median_mm": float(np.median(v["p"])), "pos_max_mm": float(np.max(v["p"])),
                  "rot_median_deg": float(np.median(v["r"])), "rot_max_deg": float(np.max(v["r"]))}
    out["by_k"] = {str(k): {**{f"{n}_median_mm": float(np.median(d[n])) for n in SIDES},
                            "n": len(d["accum"])} for k, d in sorted(by_k.items())}
    return out


def merge(per_ep: list[dict]) -> dict:
    """Worst case across episodes. 편별 최악을 남긴다 — 평균은 한 편의 실패를 숨긴다."""
    ok = [e for e in per_ep if e.get("comparisons")]
    if not ok:
        return {"episodes_scored": 0, "episodes_total": len(per_ep)}
    out = {"episodes_scored": len(ok), "episodes_total": len(per_ep),
           "comparisons": sum(e["comparisons"] for e in ok)}
    for side in SIDES:
        out[side] = {
            "pos_median_mm": float(np.median([e[side]["pos_median_mm"] for e in ok])),
            "pos_max_mm": float(np.max([e[side]["pos_max_mm"] for e in ok])),
            "rot_median_deg": float(np.median([e[side]["rot_median_deg"] for e in ok])),
            "rot_max_deg": float(np.max([e[side]["rot_max_deg"] for e in ok])),
        }
    ks = sorted({k for e in ok for k in e["by_k"]}, key=int)
    out["by_k"] = {k: {
        **{f"{n}_median_mm": float(np.median(
            [e["by_k"][k][f"{n}_median_mm"] for e in ok if k in e["by_k"]])) for n in SIDES},
        "n": sum(e["by_k"][k]["n"] for e in ok if k in e["by_k"]),
    } for k in ks}
    return out


def verdict(m: dict, tol_mm: float) -> dict:
    """Decide, and refuse to decide when the data cannot tell them apart.
    판정한다. 데이터가 둘을 구분 못 하면 판정하지 않는다."""
    if not m.get("episodes_scored"):
        return {"status": "NO_DATA", "note": "채점된 편이 0 이다. 판정 불가"}
    a = m["accum"]["pos_median_mm"]
    b = m["anchor"]["pos_median_mm"]
    dep = m["deployed"]["pos_median_mm"]
    a_ok, b_ok = a <= tol_mm, b <= tol_mm
    if b_ok and not a_ok:
        return {"status": "ANCHOR", "note":
                f"앵커가 맞다 (중앙 {b:.4f}mm). 누적은 {a:.3f}mm 다. "
                f"현행 so101_infer 공식은 {dep:.3f}mm 다. "
                "`T = T @ A` 를 `T0 @ A` 로 바꿔야 한다"}
    if a_ok and not b_ok:
        return {"status": "ACCUM", "note":
                f"누적이 맞다 (중앙 {a:.4f}mm). 앵커는 {b:.3f}mm 다. "
                f"다만 현행 so101_infer 공식은 {dep:.3f}mm 다 — 청크 안 k<{EXEC_LO} 를 "
                "버리고 곱하기 때문이다. 누적이 맞아도 현행 코드는 틀렸다"}
    if a_ok and b_ok:
        return {"status": "INDISTINGUISHABLE", "note":
                f"둘 다 {tol_mm}mm 안이다 (누적 {a:.4f} · 앵커 {b:.4f}). "
                "이 데이터는 두 해석을 구분하지 못한다. 델타가 너무 작다 — 결론으로 쓰지 마라"}
    return {"status": "NEITHER", "note":
            f"둘 다 벗어난다 (누적 {a:.3f} · 앵커 {b:.3f} mm). 제3의 규약이거나 "
            "복원 연쇄 자체가 틀렸다. 어느 쪽도 결론으로 쓰지 마라"}


def _synth(kind: str, n: int = 12, horizon: int = 8) -> np.ndarray:
    """Build action data under a known convention. 정답을 아는 합성 데이터."""
    ang = math.radians(4.0)
    step = np.eye(4)
    step[:3, :3] = np.array([[math.cos(ang), -math.sin(ang), 0.0],
                             [math.sin(ang), math.cos(ang), 0.0], [0.0, 0.0, 1.0]])
    step[:3, 3] = [0.015, -0.006, 0.003]
    truth = [np.eye(4)]
    for _ in range(n + horizon + 2):
        truth.append(truth[-1] @ step)
    act = np.zeros((n, horizon, 10))
    for i in range(n):
        for k in range(horizon):
            if kind == "anchor":
                rel = np.linalg.inv(truth[i]) @ truth[i + k + 1]      # 청크 시작 기준
            else:
                rel = np.linalg.inv(truth[i + k]) @ truth[i + k + 1]  # 한 스텝씩 (누적 규약)
            R = rel[:3, :3]
            act[i, k] = np.r_[rel[:3, 3], R[0, :], R[1, :], 0.05]
    return act


def selftest() -> int:
    """Known-answer rows for BOTH conventions, plus refusal rows.
    두 규약 각각에 정답 아는 행을 두고, 거부 행도 둔다."""
    log, bad = [], 0

    def chk(name: str, cond: bool, note: str = "") -> None:
        nonlocal bad
        log.append((name, bool(cond), note))
        if not cond:
            bad += 1

    # [1] 앵커 규약 데이터 -> ANCHOR 로 판정되어야 한다
    v1 = verdict(merge([probe_episode(_synth("anchor"))]), 1e-6)
    chk("1 앵커 데이터 -> ANCHOR", v1["status"] == "ANCHOR", v1["status"])

    # [2] 누적 규약 데이터 -> ACCUM 으로 판정되어야 한다 (반대 방향 판별행)
    v2 = verdict(merge([probe_episode(_synth("accum"))]), 1e-6)
    chk("2 누적 데이터 -> ACCUM (반대 판별행)", v2["status"] == "ACCUM", v2["status"])

    # [3] 두 합성이 실제로 다른 데이터인가 — 같으면 위 두 행이 무의미하다
    d = float(np.abs(_synth("anchor") - _synth("accum")).max())
    chk("3 두 합성이 서로 다르다 (판별행)", d > 1e-3, f"최대차 {d:.4f}")

    # [4] 델타가 0 이면 구분 불가로 거부해야 한다
    zero = np.zeros((6, 8, 10))
    zero[:, :, 3], zero[:, :, 7] = 1.0, 1.0          # 항등 회전
    v3 = verdict(merge([probe_episode(zero)]), 1e-6)
    chk("4 항등 입력 -> INDISTINGUISHABLE", v3["status"] == "INDISTINGUISHABLE", v3["status"])

    # [5] 빈 입력 -> NO_DATA. '없음'과 '괜찮음'이 같은 출력이면 안 된다
    chk("5 빈 입력 -> NO_DATA", verdict(merge([]), 1e-6)["status"] == "NO_DATA")

    # [6] 모수가 실려 나오는가
    m = merge([probe_episode(_synth("anchor"))])
    chk("6 모수 보고", m.get("comparisons", 0) > 0 and m.get("episodes_scored") == 1,
        f"비교 {m.get('comparisons')}건 · 편 {m.get('episodes_scored')}/{m.get('episodes_total')}")

    # [7] 현행 so101_infer 공식은 누적 데이터에서도 틀려야 한다 (k<lo 를 버리므로)
    m7 = merge([probe_episode(_synth("accum"))])
    chk("7 누적 데이터에서도 현행 공식은 틀린다 (판별행)",
        m7["accum"]["pos_median_mm"] < 1e-9 < m7["deployed"]["pos_median_mm"],
        f"누적 {m7['accum']['pos_median_mm']:.6f} · 현행 {m7['deployed']['pos_median_mm']:.3f} mm")

    for nm, ok, note in log:
        print(f"  {'OK ' if ok else 'FAIL'}  {nm}" + (f"   {note}" if note else ""))
    print(f"\n자체검증 {len(log) - bad}/{len(log)}")
    return 1 if bad else 0


def main() -> None:
    """CLI entry point. 명령행 진입점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", help="v10 디렉터리 (dataset.json + <편>.npz)")
    ap.add_argument("--limit", type=int, default=20, help="앞에서 N편만 (0=전부)")
    ap.add_argument("--tol-mm", type=float, default=0.01, help="0 으로 볼 위치 오차 상한")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 수치를 내지 않는다")
    if not a.dataset:
        ap.error("--dataset 이 필요하다")

    d = Path(a.dataset).expanduser()
    meta = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
    eps = sorted(meta["episodes"])
    want = eps[: a.limit] if a.limit else eps
    print(f"\n입력 {meta.get('schema')} · 전체 {len(eps)}편 · 검사 {len(want)}편 "
          f"· 실행 슬라이스 [{EXEC_LO},{EXEC_HI})\n")

    per_ep, skipped = [], []
    for name in want:
        f = d / f"{name}.npz"
        if not f.exists():
            skipped.append({"episode": name, "reason": "npz 없음"}); continue
        act = np.asarray(np.load(f)["action"], dtype=np.float64)
        if act.ndim != 3 or act.shape[2] != 10:
            skipped.append({"episode": name, "reason": f"형상 {tuple(act.shape)}"}); continue
        per_ep.append(probe_episode(act))

    m = merge(per_ep)
    print(f"채점 {m.get('episodes_scored', 0)}/{len(want)}편 · 건너뜀 {len(skipped)}편 "
          f"· 비교 {m.get('comparisons', 0)}건\n")
    if not m.get("episodes_scored"):
        raise SystemExit("!! 채점된 편이 0 이다. 수치를 내지 않는다")

    for side, tag in (("accum", "A 누적 T0@A0@..@Ak"), ("anchor", "B 앵커 T0@Ak"),
                      ("deployed", "C 현행 so101_infer")):
        s = m[side]
        print(f"{tag:<28} 위치 중앙 {s['pos_median_mm']:10.4f} mm  최대 {s['pos_max_mm']:11.4f} mm"
              f"   회전 중앙 {s['rot_median_deg']:8.4f} 도")
    print(f"\n{'k':<4}{'누적[mm]':>14}{'앵커[mm]':>14}{'현행[mm]':>14}{'비교':>8}")
    for k, v in m["by_k"].items():
        print(f"{k:<4}{v['accum_median_mm']:>14.4f}{v['anchor_median_mm']:>14.4f}"
              f"{v['deployed_median_mm']:>14.4f}{v['n']:>8}")

    v = verdict(m, a.tol_mm)
    print(f"\n판정: {v['status']}\n  {v['note']}")

    if a.out:
        doc = {"dataset": str(d), "schema": meta.get("schema"),
               "episodes_requested": len(want), "skipped": skipped,
               "exec_slice": [EXEC_LO, EXEC_HI], "tol_mm": a.tol_mm,
               "summary": m, "verdict": v,
               "limits": ["실물 미사용 · MuJoCo 미사용 · 순수 행렬 연산",
                          "복원 연쇄(k=0 누적)를 정답으로 둔다. 그것이 틀렸으면 NEITHER 가 나온다"]}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
