"""Aggregate an external SLAM+gripper batch and cross-check it against our own screen.
외부 SLAM·그리퍼 배치 산출물을 전수 집계하고 우리 선별 결과와 대조한다.

왜 (2026-09-18)
---------------
현석이 우리 0918 수집분 92편을 자기 ORB-SLAM3 파이프라인에 통째로 돌려서 줬다.
**같은 입력에 독립 구현**이라 우리 계측기의 하한성을 직접 검정할 수 있다.

표본 1편 대조 (rec_1789701868830_2c55035d) 🟢
    두 마커 검출률       우리 0.650  vs  현석 0.847
    최장 연속 미검출     우리 18프레임 vs 현석 5프레임

⚠️ **이 도구는 게이트를 두지 않는다.** 검출률·추적률 임계를 본 적이 없다.
   분포를 먼저 내고 트랙 A 와 같이 정한다. 임의로 박으면 계측이 아니라 취향이다.

⚠️ 수치는 현석 파이프라인 것이다. 캘리브레이션·좌표계가 트랙 A v10 기준과 같다는
   보장이 없다. **성능 근거로 인용하지 마라.** 이 도구가 답하는 것은
   "SLAM 이 붙었나 · gap 이 뽑혔나 · 우리 선별과 일치하나" 셋뿐이다.

읽는 파일 (에피소드마다)
    prepare_report.json          frame_count · imu · gripper 내역
    gripper_report.json          detection_rate · maximum_missing_run_frames · width_*
    slam/trajectory_validation.json  status · tracked_ratio · loss_episodes · keyframes
    slam/receipt.json            map/mask/settings/binary SHA-256 (재현성)

계측기 규칙
    모수를 같이 찍는다 — 찾은 편 / 전체 디렉터리
    검색 0건이면 범위가 완전한지 먼저 본다
    정답을 아는 행을 자체검증에 넣는다

Usage
-----
  python summarize_slam_batch.py --selftest
  python summarize_slam_batch.py --root <처리루트> --ours quality_0918.json --out slam_batch.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path


# ── 순수 계산 — 자체검증 대상 ─────────────────────────────────────────────

def quantiles(v: list[float]) -> dict:
    """Min / q25 / median / q75 / max with n. 분위수와 표본 수."""
    if not v:
        return {"n": 0}
    s = sorted(v)
    q = lambda f: s[min(len(s) - 1, int(f * (len(s) - 1)))]          # noqa: E731
    return {"n": len(s), "min": s[0], "q25": q(.25), "median": q(.5),
            "q75": q(.75), "max": s[-1]}


def compare_sets(ours: set[str], theirs: set[str]) -> dict:
    """Episode-name agreement. 편 명단 일치 여부. 양쪽 모수를 같이 낸다."""
    return {"both": len(ours & theirs), "ours_only": sorted(ours - theirs)[:10],
            "theirs_only": sorted(theirs - ours)[:10],
            "n_ours": len(ours), "n_theirs": len(theirs),
            "identical": ours == theirs}


def selftest() -> int:
    bad = 0
    q = quantiles([1, 2, 3, 4, 5])
    ok = q["n"] == 5 and q["min"] == 1 and q["median"] == 3 and q["max"] == 5
    print(f"[1] 분위수 → 중앙 {q['median']} n {q['n']}  기대 3 / 5  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    q = quantiles([])
    ok = q == {"n": 0}
    print(f"[2] 빈 입력 → {q}  기대 n 0 (0.0 아님)  ", end="")
    print("OK" if ok else "!! 실패 — 빈 것과 0 이 같은 출력"); bad += (not ok)

    c = compare_sets({"a", "b"}, {"a", "b"})
    ok = c["identical"] and c["both"] == 2 and c["n_ours"] == 2
    print(f"[3] 동일 명단 → 일치 {c['identical']} · 공통 {c['both']}/{c['n_ours']}  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    c = compare_sets({"a", "b"}, {"a", "c"})
    ok = (not c["identical"]) and c["ours_only"] == ["b"] and c["theirs_only"] == ["c"]
    print(f"[4] 다른 명단 → 우리만 {c['ours_only']} 상대만 {c['theirs_only']}  ", end="")
    print("OK" if ok else "!! 실패 — 차집합을 못 잡는다"); bad += (not ok)

    c = compare_sets(set(), set())
    ok = c["identical"] and c["both"] == 0 and c["n_ours"] == 0
    print(f"[5] 양쪽 빈 명단 → 일치라고 나오지만 모수 {c['n_ours']}/{c['n_theirs']}  ", end="")
    print("OK" if ok else "!! 실패")
    print("     ⚠️ 모수 0 을 같이 찍어야 '0개 불일치'와 '0개 비교'가 구분된다")
    bad += (not ok)

    # 정답을 아는 행: read_episode 가 붙이는 접두어를 dist 키가 그대로 쓰는가.
    # 2026-09-18 에 g_ 접두어를 빠뜨려 "값 없음 n=0/92" 가 나왔다. 모수를 찍어서 드러났다.
    import inspect
    src = inspect.getsource(main)
    keys = [l.split('dist("')[1].split('"')[0] for l in src.splitlines() if 'dist("' in l]
    gripper_fields = {"detection_rate", "maximum_missing_run_frames", "width_median_mm",
                      "width_min_mm", "width_max_mm", "raw_closed_marker_center_mm"}
    wrong = [k for k in keys if k in gripper_fields]
    ok = not wrong
    print(f"[6] 그리퍼 키에 g_ 접두어 누락 → {len(wrong)}개  기대 0  ", end="")
    print("OK" if ok else f"!! 실패 — {wrong}"); bad += (not ok)

    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


# ── 집계 ─────────────────────────────────────────────────────────────────

def read_episode(d: Path) -> dict:
    """Gather one episode's reports. 편 하나의 보고서를 모은다. 없는 건 None."""
    def js(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                            # noqa: BLE001
            return None
    prep, grip = js(d / "prepare_report.json"), js(d / "gripper_report.json")
    val, rec = js(d / "slam" / "trajectory_validation.json"), js(d / "slam" / "receipt.json")
    r = {"episode": d.name,
         "has_prepare": prep is not None, "has_gripper": grip is not None,
         "has_validation": val is not None, "has_receipt": rec is not None,
         "has_trajectory_csv": (d / "slam" / "camera_trajectory.csv").exists()}
    if prep:
        r["outcome"] = prep.get("outcome"); r["prepare_status"] = prep.get("status")
        r["frames"] = prep.get("frame_count")
        r["imu_shift_ms"] = (prep.get("telemetry") or {}).get("applied_imu_shift_ms")
    if grip:
        for k in ("status", "detection_rate", "maximum_missing_run_frames",
                  "width_median_mm", "width_min_mm", "width_max_mm",
                  "raw_closed_marker_center_mm", "decoded_frames",
                  "frames_with_both_markers"):
            r[f"g_{k}"] = grip.get(k)
    if val:
        for k in ("status", "frames", "duration_s", "tracked_frames",
                  "tracked_ratio_after_initialization",
                  "loss_episodes_after_initialization", "keyframes", "path_length_m"):
            r[f"v_{k}"] = val.get(k)
        ext = val.get("position_extent_m")
        if ext:
            r["v_extent_max_m"] = max(ext)
    if rec:
        r["receipt"] = {k: (v[:12] if isinstance(v, str) else v) for k, v in rec.items()}
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", help="processed/ 디렉터리")
    ap.add_argument("--ours", help="우리 quality_*.json (pass 배열 대조용)")
    ap.add_argument("--object-width-mm", type=float, default=41.0)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 판정하지 않는다")
    print()

    root = Path(a.root).expanduser()
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    eps = [p for p in dirs if (p / "prepare_report.json").exists()]
    print(f"편 {len(eps)} / 디렉터리 {len(dirs)}  (루트 {root})")
    if not eps:
        raise SystemExit("!! 0편. **없는 게 아니라 못 찾은 것일 수 있다** — "
                         "루트가 processed/ 를 가리키는지 확인하라")

    rows = [read_episode(d) for d in eps]

    miss = {k: sum(1 for r in rows if not r[k])
            for k in ("has_prepare", "has_gripper", "has_validation",
                      "has_receipt", "has_trajectory_csv")}
    print("\n=== 산출물 완비 (없는 편 수 / 전체) ===")
    for k, v in miss.items():
        print(f"  {k:22s} 누락 {v} / {len(rows)}")

    def cnt(key: str) -> None:
        c: dict = {}
        for r in rows:
            c[r.get(key)] = c.get(r.get(key), 0) + 1
        print(f"  {key:26s} " + " · ".join(f"{k}={v}" for k, v in sorted(c.items(), key=str)))

    print("\n=== 상태 ===")
    cnt("prepare_status"); cnt("g_status"); cnt("v_status"); cnt("outcome")

    def dist(key: str, label: str, fmt: str = "{:.3f}") -> None:
        q = quantiles([r[key] for r in rows if r.get(key) is not None])
        if q["n"] == 0:
            print(f"  {label:26s} 값 없음 (n=0 / {len(rows)})"); return
        print(f"  {label:26s} 최소 {fmt.format(q['min'])} · 25% {fmt.format(q['q25'])} · "
              f"중앙 {fmt.format(q['median'])} · 75% {fmt.format(q['q75'])} · "
              f"최대 {fmt.format(q['max'])}   n={q['n']}/{len(rows)}")

    # 우리 합격 ∩ SLAM pass = 실제 학습 가능분. 이게 상한이다
    print("\n=== SLAM 분포 — 임계 없음, 관찰만 ===")
    dist("v_tracked_ratio_after_initialization", "추적률(초기화 후)")
    dist("v_loss_episodes_after_initialization", "추적 끊김 횟수", "{:.0f}")
    dist("v_keyframes", "키프레임", "{:.0f}")
    dist("v_path_length_m", "궤적 길이 [m]")
    dist("v_extent_max_m", "위치 범위 최대축 [m]")
    dist("v_duration_s", "길이 [s]", "{:.2f}")

    print("\n=== 그리퍼 분포 — 임계 없음, 관찰만 ===")
    dist("g_detection_rate", "두 마커 검출률")
    dist("g_maximum_missing_run_frames", "최장 연속 미검출 [프레임]", "{:.0f}")
    dist("g_width_median_mm", "gap 중앙 [mm]", "{:.2f}")
    dist("g_width_min_mm", "gap 최소 [mm]", "{:.2f}")
    dist("g_width_max_mm", "gap 최대 [mm]", "{:.2f}")
    dist("g_raw_closed_marker_center_mm", "폐쇄 기준 마커거리 [mm]", "{:.2f}")
    dist("imu_shift_ms", "적용된 IMU 시프트 [ms]", "{:.2f}")

    wmin = [r["g_width_min_mm"] for r in rows if r.get("g_width_min_mm") is not None]
    if wmin:
        med = st.median(wmin)
        d = abs(med - a.object_width_mm)
        print(f"\n물체 실폭 대조   편별 gap 최소의 중앙 {med:.2f} mm vs 물체 {a.object_width_mm:.2f} mm "
              f"→ 차이 {d:.2f} mm")
        print("  ⚠️ 이건 진단이다. 학습 투입 판정은 gap 채널 계측기(check_gap_validity)가 한다")

    cross = None
    if a.ours:
        ours = json.loads(Path(a.ours).expanduser().read_text(encoding="utf-8"))
        cross = compare_sets(set(ours.get("pass", [])), {r["episode"] for r in rows})
        print(f"\n=== 우리 선별과 대조 ===")
        print(f"  공통 {cross['both']} · 우리 합격 {cross['n_ours']} · 상대 처리 {cross['n_theirs']}")
        if cross["ours_only"]:
            print(f"  우리 합격인데 상대에 없음: {cross['ours_only']}")
        if cross["theirs_only"]:
            print(f"  상대에 있는데 우리 불합격: {cross['theirs_only']}")
            print("  ↑ 우리가 뺀 편이다. 상대가 처리했어도 학습에 넣지 않는다")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"root": str(root), "episodes": len(rows), "dirs": len(dirs),
             "object_width_mm": a.object_width_mm, "missing": miss,
             "cross_check_with_ours": cross, "rows": rows},
            indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
