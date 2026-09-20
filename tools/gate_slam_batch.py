#!/usr/bin/env python3
"""Judge a SLAM batch before building any zarr.
SLAM 배치를 zarr 만들기 전에 판정한다.

왜 필요한가
-----------
게이트가 raw(촬영) 와 zarr(변환 후) 두 곳에만 있으면 그 사이가 빈다.
SLAM 이 반쯤 깨진 편이 섞여 들어가면 변환까지 하고 나서야 알게 된다.

읽는 것 (전부 processed/rec_*/ 안, SLAM 산출물)
    slam/trajectory_validation.json   추적률 · 손실 · 키프레임 · 경로길이 · 위치 범위
    slam/receipt.json                 load_map_sha256 — **어느 지도를 썼나**
    gripper_report.json               마커 검출률 · 결손 구간 · 개구 중앙/최대
    slam/camera_trajectory.csv        행수 대조

**배치 전체에서 load_map_sha256 이 하나여야 한다.** 서로 다른 아틀라스를 쓴 편이
섞이면 좌표계가 섞이고, 그건 어떤 지표에도 안 나타난다.

Usage
    python gate_slam_batch.py --selftest
    python gate_slam_batch.py --processed ~/mnt/s22_umi_v3/processed --out out/slam_gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ── 게이트 (배치 돌리기 전에 박는다) ────────────────────────────────────────
# 참조: 현석 dataset_config (maximum_lost_frames 10 · marker_missing_run 15 · gap<=0.09m)
#       SO-101 도달 반경 0.3m · 현석 s22 편당 이동거리 0.371m
GATES = {
    "tracked_ratio_min":    {"v": 0.98, "why": "초기화 후 추적률. 현석 배치는 1.0 이었다"},
    "loss_episodes_max":    {"v": 10,   "why": "현석 config maximum_lost_frames_after_initialization"},
    "keyframes_min":        {"v": 5,    "why": "키프레임이 너무 적으면 궤적이 신뢰 안 된다"},
    "path_length_max_m":    {"v": 1.00, "why": "현석 s22 편당 0.371m · v4 1.416m 는 길다"},
    "extent_max_m":         {"v": 0.30, "why": "SO-101 도달 반경. 카메라 궤적 기준 대리값이다(TCP 아님)"},
    "detection_rate_min":   {"v": 0.80, "why": "마커 검출률. 현석 v4 0.847"},
    "missing_run_max":      {"v": 15,   "why": "현석 config maximum_marker_missing_run_frames"},
    "width_max_mm":         {"v": 90.0, "why": "현석 config maximum_gripper_width_m 0.09"},
    # ⚠️ 2026-09-21 정정 — 초판은 width_median_mm(편 전체 중앙)을 봤다. 틀렸다.
    #    그 값은 벌린 상태가 지배해서 92편 중 77편(84%)을 떨어뜨렸다. 데이터가 아니라
    #    기준이 틀린 것이었다. 파지 순간의 대리값은 **편 최소 개구**다.
    "width_min_lo":         {"v": 30.0, "why": "파지 순간 개구 하한. 물체 41mm"},
    "width_min_hi":         {"v": 55.0, "why": "상한. 실측 예 37.7mm"},
}


def load(p: Path):
    """Read json, None if absent. 없으면 None — '없음'과 '괜찮음'을 가른다."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return None


def measure(d: Path) -> dict:
    """Measure one processed episode. 처리된 편 하나를 잰다."""
    r: dict = {"episode": d.name, "missing": []}
    tv = load(d / "slam" / "trajectory_validation.json")
    if tv:
        r.update({
            "traj_status": tv.get("status"),
            "frames": tv.get("frames"),
            "duration_s": tv.get("duration_s"),
            "tracked_ratio": tv.get("tracked_ratio_after_initialization"),
            "loss_episodes": tv.get("loss_episodes_after_initialization"),
            "keyframes": tv.get("keyframes"),
            "path_length_m": tv.get("path_length_m"),
            "extent_max_m": max(tv["position_extent_m"]) if tv.get("position_extent_m") else None,
        })
    else:
        r["missing"].append("trajectory_validation.json")

    rc = load(d / "slam" / "receipt.json")
    r["load_map_sha256"] = rc.get("load_map_sha256") if rc else None
    if not rc:
        r["missing"].append("receipt.json")

    gr = load(d / "gripper_report.json")
    if gr:
        r.update({
            "grip_status": gr.get("status"),
            "detection_rate": gr.get("detection_rate"),
            "missing_run": gr.get("maximum_missing_run_frames"),
            "width_median_mm": gr.get("width_median_mm"),
            "width_min_mm": gr.get("width_min_mm"),
            "width_max_mm": gr.get("width_max_mm"),
        })
    else:
        r["missing"].append("gripper_report.json")

    ct = d / "slam" / "camera_trajectory.csv"
    if ct.exists():
        with ct.open(encoding="utf-8") as f:
            r["traj_rows"] = sum(1 for _ in f) - 1
    else:
        r["missing"].append("camera_trajectory.csv")
    return r


def judge(r: dict, gates: dict | None = None) -> dict:
    """Apply gates. 없는 값은 미판정이지 통과가 아니다."""
    g = {k: v["v"] for k, v in (gates or GATES).items()}
    rows = []

    # ⚠️ 2026-09-21 정정 — 초판은 시연 길이(이동거리·궤적 범위)를 필수로 걸어
    #    92편 전부를 떨어뜨렸다. 그런데 현석은 **바로 그 데이터로 v4 를 학습시켰고
    #    돌아갔다.** 긴 궤적은 재현 난이도이지 폐기 사유가 아니다.
    #    프로젝트 기조 — "결과 모방이지 궤적 모방이 아니다."
    #    필수(SLAM 품질) 와 권고(시연 스타일) 를 가른다. 판정은 필수만 본다.
    SOFT = {"편당 이동거리 m", "카메라 궤적 범위 m"}

    def row(n, ok, got, want):
        rows.append({"name": n, "ok": ok, "got": got, "want": want,
                     "tier": "권고" if n in SOFT else "필수"})

    def over(n, v, lim, f="{:.3f}"):
        row(n, None if v is None else v >= lim, "없음 — 대조 불가" if v is None else f.format(v), f">= {lim}")

    def under(n, v, lim, f="{:.3f}"):
        row(n, None if v is None else v <= lim, "없음 — 대조 불가" if v is None else f.format(v), f"<= {lim}")

    row("궤적 판정", r.get("traj_status") == "pass", r.get("traj_status"), "pass")
    row("그리퍼 판정", r.get("grip_status") == "pass", r.get("grip_status"), "pass")
    over("초기화 후 추적률", r.get("tracked_ratio"), g["tracked_ratio_min"])
    under("추적 손실 구간", r.get("loss_episodes"), g["loss_episodes_max"], "{:.0f}")
    over("키프레임 수", r.get("keyframes"), g["keyframes_min"], "{:.0f}")
    under("편당 이동거리 m", r.get("path_length_m"), g["path_length_max_m"])
    under("카메라 궤적 범위 m", r.get("extent_max_m"), g["extent_max_m"])
    over("마커 검출률", r.get("detection_rate"), g["detection_rate_min"])
    under("마커 결손 최대 구간", r.get("missing_run"), g["missing_run_max"], "{:.0f}")
    under("개구 최대 mm", r.get("width_max_mm"), g["width_max_mm"], "{:.1f}")
    wm = r.get("width_min_mm")
    row("파지 개구(편 최소) mm",
        None if wm is None else g["width_min_lo"] <= wm <= g["width_min_hi"],
        "없음 — 대조 불가" if wm is None else f"{wm:.1f}",
        f"{g['width_min_lo']}~{g['width_min_hi']}")
    fr, tr = r.get("frames"), r.get("traj_rows")
    row("궤적 행수 = 프레임 수", None if (fr is None or tr is None) else fr == tr,
        f"{tr} / {fr}" if tr is not None else "없음 — 대조 불가", "같을 것")

    hard = [x for x in rows if x["tier"] == "필수"]
    soft = [x for x in rows if x["tier"] == "권고"]
    p = sum(1 for x in hard if x["ok"] is True)
    f = sum(1 for x in hard if x["ok"] is False)
    u = sum(1 for x in hard if x["ok"] is None)
    warn = sum(1 for x in soft if x["ok"] is False)
    return {"rows": rows, "passed": p, "failed": f, "unknown": u, "total": len(hard),
            "warnings": warn, "soft_total": len(soft),
            "verdict": "PASS" if f == 0 and u == 0 else ("FAIL" if f else "INCOMPLETE")}


def batch_checks(docs: list[dict]) -> dict:
    """Checks that only make sense across the whole batch. 배치 전체에서만 보이는 것."""
    maps = Counter(d.get("load_map_sha256") for d in docs if d.get("load_map_sha256"))
    unknown = sum(1 for d in docs if not d.get("load_map_sha256"))
    rows = [{
        "name": "배치가 같은 지도를 썼다",
        "ok": None if unknown else (len(maps) == 1),
        "got": (f"지도 {len(maps)}종 · 영수증 없음 {unknown} / 전체 {len(docs)}"
                + ("  " + " · ".join(f"{k[:8]}×{v}" for k, v in maps.most_common(3)) if maps else "")),
        "want": "1종",
    }]
    return {"rows": rows, "maps": {k: v for k, v in maps.items()}, "receipts_missing": unknown}


def _synth(tmp: Path, name: str, *, ok: bool = True, map_sha: str = "a" * 64,
           lost: int = 0, det: float = 0.85, extent: float = 0.25) -> Path:
    """Known-answer processed folder. 정답을 아는 처리 폴더."""
    d = tmp / name
    (d / "slam").mkdir(parents=True, exist_ok=True)
    (d / "slam" / "trajectory_validation.json").write_text(json.dumps({
        "status": "pass" if ok else "fail", "frames": 249, "duration_s": 8.3,
        "tracked_frames": 249, "tracked_ratio_after_initialization": 1.0 if ok else 0.5,
        "loss_episodes_after_initialization": lost, "keyframes": 22,
        "path_length_m": 0.45, "position_extent_m": [extent, 0.23, 0.29]}), encoding="utf-8")
    (d / "slam" / "receipt.json").write_text(json.dumps({"load_map_sha256": map_sha}), encoding="utf-8")
    (d / "gripper_report.json").write_text(json.dumps({
        "status": "pass", "detection_rate": det, "maximum_missing_run_frames": 5,
        "width_median_mm": 41.0, "width_min_mm": 37.7, "width_max_mm": 70.6}), encoding="utf-8")
    with (d / "slam" / "camera_trajectory.csv").open("w", encoding="utf-8") as f:
        f.write("frame_idx,timestamp\n")
        for i in range(249):
            f.write(f"{i},{i / 30:.6f}\n")
    return d


def selftest() -> int:
    """Known-answer and discriminating rows. 정답 아는 행과 판별행."""
    import tempfile
    log, bad = [], 0

    def chk(n, c, note=""):
        nonlocal bad
        log.append((n, bool(c), note))
        if not c:
            bad += 1

    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        good = judge(measure(_synth(t, "good")))
        chk("1 정상 편 -> PASS", good["verdict"] == "PASS", f"{good['passed']}/{good['total']}")
        chk("2 검사 항목 수 (필수 10 · 권고 2)",
            good["total"] == 10 and good["soft_total"] == 2,
            f"필수 {good['total']} · 권고 {good['soft_total']}")
        chk("3 추적 실패 -> FAIL (판별행)",
            judge(measure(_synth(t, "lost", ok=False)))["verdict"] == "FAIL")
        chk("4 손실 구간 초과 -> FAIL (판별행)",
            judge(measure(_synth(t, "many", lost=30)))["verdict"] == "FAIL")
        chk("5 검출률 미달 -> FAIL (판별행)",
            judge(measure(_synth(t, "lowdet", det=0.5)))["verdict"] == "FAIL")
        # 시연이 크면 **경고**다. 폐기가 아니다 — 현석은 그런 데이터로 학습에 성공했다
        far = judge(measure(_synth(t, "far", extent=0.52)))
        chk("6 도달 범위 초과 -> 경고, 판정은 PASS (판별행)",
            far["verdict"] == "PASS" and far["warnings"] >= 1,
            f"{far['verdict']} · 경고 {far['warnings']}/{far['soft_total']}")
        near = judge(measure(_synth(t, "near", extent=0.25)))
        chk("6b 범위 안이면 경고 0 (반대 판별행)", near["warnings"] == 0,
            f"경고 {near['warnings']}")
        empty = t / "empty"; (empty / "slam").mkdir(parents=True)
        j = judge(measure(empty))
        chk("7 빈 폴더 -> 통과 0 · 미판정", j["passed"] == 0 and j["verdict"] != "PASS",
            f"통과 {j['passed']} · 미판정 {j['unknown']} · {j['verdict']}")

        same = [measure(_synth(t, f"s{i}")) for i in range(3)]
        chk("8 같은 지도 3편 -> 통과", batch_checks(same)["rows"][0]["ok"] is True)
        mixed = same + [measure(_synth(t, "other", map_sha="b" * 64))]
        b = batch_checks(mixed)
        chk("9 지도 섞이면 잡는다 (판별행)", b["rows"][0]["ok"] is False, b["rows"][0]["got"])
        noR = measure(_synth(t, "norec")); noR["load_map_sha256"] = None
        chk("10 영수증 없으면 미판정 (통과 아님)",
            batch_checks([noR])["rows"][0]["ok"] is None)

    for nm, ok, note in log:
        print(f"  {'OK ' if ok else 'FAIL'}  {nm}" + (f"   {note}" if note else ""))
    print(f"\n자체검증 {len(log) - bad}/{len(log)}")
    return 1 if bad else 0


def main() -> None:
    """CLI entry point. 명령행 진입점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--processed", help="rec_* 처리 결과가 들어 있는 디렉터리")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 수치를 내지 않는다")
    if not a.processed:
        ap.error("--processed 가 필요하다")

    root = Path(a.processed).expanduser()
    eps = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("rec_"))
    if not eps:
        raise SystemExit(f"!! {root} 안에 rec_* 가 없다. 경로가 맞는지 먼저 봐라")

    docs, tally = [], Counter()
    for d in eps:
        r = measure(d)
        j = judge(r)
        r["gates"] = j
        docs.append(r)
        tally[j["verdict"]] += 1
        if j["verdict"] != "PASS":
            why = " · ".join(f"{x['name']} {x['got']}" for x in j["rows"]
                             if x["ok"] is not True and x["tier"] == "필수")
            print(f"[{ {'FAIL': '불합격', 'INCOMPLETE': '미판정'}[j['verdict']]:^4}] {d.name:<34} {why}")

    b = batch_checks(docs)
    n = len(eps)
    # 항목별 분해 — 전부 불합격일 때 '게이트가 틀렸나'를 가르는 유일한 방법이다
    per = Counter()
    unk = Counter()
    for d in docs:
        for x in d["gates"]["rows"]:
            if x["ok"] is False:
                per[x["name"]] += 1
            elif x["ok"] is None:
                unk[x["name"]] += 1
    print(f"\n통과 {tally['PASS']} · 불합격 {tally['FAIL']} · 미판정 {tally['INCOMPLETE']} / 전체 {n}")
    warned = sum(1 for d in docs if d["gates"]["warnings"])
    print(f"권고 경고가 붙은 편 {warned} / {n}  (재현 난이도 경고이지 폐기 사유가 아니다)")
    print("\n항목별 불합격 수 (분모 전부 %d편)" % n)
    for k, v in per.most_common():
        tier = "권고" if k in {"편당 이동거리 m", "카메라 궤적 범위 m"} else "필수"
        print(f"  [{tier}] {k:<22} {v:>4} 편  ({v / n:.0%})")
    if unk:
        print("항목별 미판정 수")
        for k, v in unk.most_common():
            print(f"  {k:<22} {v:>4} 편")
    if per and max(per.values()) == n:
        print("\n** 한 항목이 전 편을 떨어뜨린다. 데이터가 아니라 그 기준을 먼저 의심해라 **")
    for x in b["rows"]:
        mark = {True: "통과", False: "불합격", None: "미판정"}[x["ok"]]
        print(f"[{mark:^4}] {x['name']:<24} {x['got']}   기준 {x['want']}")
    if tally["PASS"] < n or any(x["ok"] is not True for x in b["rows"]):
        print("** 미판정은 통과가 아니다. 불합격 편은 빼고 가되, 지도가 섞였으면 전부 다시 돌린다 **")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"processed": str(root), "episodes": docs, "batch": b,
             "tally": dict(tally), "total": n, "gates": GATES},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
