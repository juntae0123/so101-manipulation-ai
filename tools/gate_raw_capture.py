#!/usr/bin/env python3
"""Judge a raw UMI capture the moment it is shot, before SLAM.
촬영 직후 raw 만 보고 판정한다. SLAM·변환을 기다리지 않는다.

왜 필요한가
-----------
기존 게이트(audit_umi_zarr)는 zarr 단계다. 잘못 찍은 걸 SLAM 돌리고 변환까지 한
뒤에야 알게 된다. 현장에서 20편 찍고 3시간 뒤 "다시 찍어야 함"이 나오면 끝이다.
**이 도구는 촬영 폴더만 보고 몇 초 안에 답한다.**

읽는 것 (전부 raw/rec_*/ 안)
    manifest.json      outcome · status · 해상도 · 방향 규약 · 마커 검증 여부
    frames.csv         프레임 수 · 표본율 · 지터 · 노출/ISO 변동 · AF 고정
    accelerometer.csv  IMU 실제 Hz · 프레임 구간 덮는가
    gyroscope.csv      동일
    encoded.csv        인코더 pts 와 센서 timestamp 정합

측정 원칙
    - 모수를 같이 찍는다 (통과 N / 전체 M)
    - 못 읽은 항목은 **미판정**이다. 통과가 아니다
    - 게이트는 촬영 전에 박는다. 결과 보고 옮기지 않는다

Usage
    python gate_raw_capture.py --selftest
    python gate_raw_capture.py --raw ~/raw/raw --out out/capture_gate.json
    python gate_raw_capture.py --raw ~/raw/raw --episode rec_1789701868830_2c55035d
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

NS = 1e9

# ── 게이트 (촬영 전에 박는다) ───────────────────────────────────────────────
# 참조 집단: 현석 s22_pick_v3 35편 (120 epoch 학습이 돌아간 배치).
#   편 길이 73~137 프레임 @30Hz = 2.4~4.6초 · 표본율 30.0021Hz
GATES = {
    "frames_min":        {"v": 60,   "why": "30Hz 에서 2.0초. 접근이 안 담긴다"},
    # ⚠️ 권고다. 폐기 사유가 아니다. 🟢 2026-09-21 — 필수로 뒀더니 92/92 전 편 불합격이
    #    났는데, 바로 그 92편으로 현석 v4 학습이 돌았다. 이 프로젝트의 기조가
    #    "결과 모방이지 궤적 모방이 아니다" 이므로 **시연 길이는 난이도 경고**로 둔다.
    #    (같은 오류를 gate_slam_batch.py 에서 하루 전에 이미 고쳤다. 옮겨오지 못했다)
    "frames_max":        {"v": 150,  "why": "권고. 30Hz 에서 5.0초. 현석 s22_pick_v3 73~137, v4 165~300"},
    "fps_lo":            {"v": 29.0, "why": "현석 30.0021Hz"},
    "fps_hi":            {"v": 31.0, "why": "동일"},
    "gap_ratio_max":     {"v": 2.5,  "why": "최대 프레임 간격 / 중앙. 2.5배 넘으면 드롭이다"},
    "imu_hz_min":        {"v": 150.0,"why": "요청 200Hz. 150 밑이면 SLAM 초기화가 흔들린다"},
    "imu_margin_s_min":  {"v": 0.0,  "why": "IMU 가 프레임 구간을 앞뒤로 덮어야 한다"},
    "exposure_cv_max":   {"v": 0.35, "why": "노출 변동계수. 조명이 흔들리면 SLAM·학습 둘 다 샌다"},
    "focus_span_max":    {"v": 0.0,  "why": "AF 가 움직이면 초점거리가 바뀐다. 고정 촬영이어야 한다"},
    "pts_skew_ms_max":   {"v": 2.0,  "why": "인코더 pts 와 센서 timestamp 차. 현석 설정도 2.0ms"},
}
REQUIRED_OUTCOME = "success"
REQUIRED_STATUS = "recorded"
REQUIRED_ORIENT = "upright_v1"


def read_csv(p: Path) -> list[dict]:
    """Read a csv into dicts. csv 를 딕트 목록으로. 없으면 빈 목록."""
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


COLUMN_DROPS: dict[str, str] = {}      # 열 이름 -> "남은 행/전체 행". 모수를 잃지 않는다


def col(rows: list[dict], name: str) -> list[float]:
    """One numeric column. 숫자 열 하나. **버린 행 수를 반드시 남긴다.**

    ⚠️ 초판은 파싱 실패를 조용히 삼켰다. 열 이름이 틀리면 전부 버려지고 빈 목록이
    나오는데, 그게 "값이 없다"와 같은 출력이라 열 오타가 미판정으로 둔갑한다.
    """
    out, dropped = [], 0
    for r in rows:
        try:
            out.append(float(r[name]))
        except (KeyError, TypeError, ValueError):
            dropped += 1
    if rows:
        COLUMN_DROPS[name] = f"{len(out)}/{len(rows)}"
        if not out:
            print(f"  ⚠️ 열 '{name}' 에서 {len(rows)}행 전부 파싱 실패 — "
                  f"열 이름이 틀렸거나 형식이 바뀌었다. 빈 값으로 넘어가지 마라",
                  file=sys.stderr)
    return out


def cv(xs: list[float]) -> float:
    """Coefficient of variation. 변동계수 — 평균이 0 이면 nan."""
    if len(xs) < 2:
        return float("nan")
    m = st.fmean(xs)
    return float("nan") if abs(m) < 1e-12 else st.pstdev(xs) / abs(m)


def measure_episode(d: Path) -> dict:
    """Measure one capture folder. 촬영 폴더 하나를 잰다. 못 읽으면 None 으로 남긴다."""
    r: dict = {"episode": d.name, "missing": []}
    mp = d / "manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        r["outcome"] = m.get("outcome")
        r["status"] = m.get("status")
        r["orientation_contract"] = m.get("image_orientation_contract")
        r["rotation_baked"] = m.get("rotation_baked_into_pixels")
        r["marker_verified"] = m.get("marker_black_square_size_verified")
        r["training_ready"] = m.get("training_ready")
        r["resolution"] = [m.get("width"), m.get("height")]
        r["device"] = m.get("device")
        r["task"] = m.get("task")
        s, e = m.get("start_elapsed_ns"), m.get("stop_elapsed_ns")
        r["record_s"] = (e - s) / NS if isinstance(s, int) and isinstance(e, int) else None
    else:
        r["missing"].append("manifest.json")

    fr = read_csv(d / "frames.csv")
    ts = sorted(col(fr, "sensor_timestamp_ns"))
    r["frames"] = len(ts)
    if len(ts) >= 3:
        dt = [(b - a) / NS for a, b in zip(ts, ts[1:])]
        med = st.median(dt)
        r["fps_median"] = 1.0 / med if med > 0 else None
        r["gap_ratio_max"] = max(dt) / med if med > 0 else None
        r["span_s"] = (ts[-1] - ts[0]) / NS
        r["exposure_cv"] = cv(col(fr, "exposure_ns"))
        r["iso_cv"] = cv(col(fr, "iso"))
        f = col(fr, "focus_diopters")
        r["focus_span"] = (max(f) - min(f)) if f else None
    else:
        r["missing"].append("frames.csv")

    for tag, fn in (("accel", "accelerometer.csv"), ("gyro", "gyroscope.csv")):
        rows = read_csv(d / fn)
        t = sorted(col(rows, "timestamp_ns"))
        if len(t) >= 3 and len(ts) >= 2:
            dur = (t[-1] - t[0]) / NS
            r[f"{tag}_hz"] = (len(t) - 1) / dur if dur > 0 else None
            # IMU 가 프레임 구간을 앞뒤로 덮는 여유[초]. 음수면 못 덮는다
            r[f"{tag}_margin_s"] = min((ts[0] - t[0]) / NS, (t[-1] - ts[-1]) / NS)
        else:
            r["missing"].append(fn)

    enc = read_csv(d / "encoded.csv")
    pts = col(enc, "pts_us")
    if pts and len(ts) >= 1 and len(pts) == len(ts):
        # encoded.pts_us 는 sensor_timestamp_ns 를 us 로 자른 값이어야 한다
        skew = [abs(p * 1000.0 - t) / 1e6 for p, t in zip(sorted(pts), ts)]
        r["pts_skew_ms_max"] = max(skew)
    elif pts:
        r["pts_skew_ms_max"] = None
        r["missing"].append(f"encoded.csv 행수 {len(pts)} != frames {len(ts)}")
    else:
        r["missing"].append("encoded.csv")

    r["video_bytes"] = (d / "video.mp4").stat().st_size if (d / "video.mp4").exists() else None
    if r["video_bytes"] is None:
        r["missing"].append("video.mp4")
    return r


UNKNOWN = object()          # 키 자체가 없을 때. None(=없음) 과 구분한다


def judge(r: dict, gates: dict | None = None) -> dict:
    """Apply gates. 게이트를 건다. 값이 없으면 미판정이지 통과가 아니다."""
    g = {k: v["v"] for k, v in (gates or GATES).items()}
    rows = []

    def row(name, ok, got, want, soft=False):
        rows.append({"name": name, "ok": ok, "got": got, "want": want, "soft": soft})

    def band(name, val, lo, hi, fmt="{:.3f}", soft=False):
        if val is None:
            row(name, None, "없음 — 대조 불가", f"{lo}~{hi}", soft)
        else:
            row(name, lo <= val <= hi, fmt.format(val), f"{lo}~{hi}", soft)

    def under(name, val, lim, fmt="{:.3f}", soft=False):
        if val is None:
            row(name, None, "없음 — 대조 불가", f"<= {lim}", soft)
        else:
            row(name, val <= lim, fmt.format(val), f"<= {lim}", soft)

    def over(name, val, lim, fmt="{:.3f}", soft=False):
        if val is None:
            row(name, None, "없음 — 대조 불가", f">= {lim}", soft)
        else:
            row(name, val >= lim, fmt.format(val), f">= {lim}", soft)

    row("촬영 성공 표기", r.get("outcome") == REQUIRED_OUTCOME, r.get("outcome"), REQUIRED_OUTCOME)
    row("녹화 상태", r.get("status") == REQUIRED_STATUS, r.get("status"), REQUIRED_STATUS)
    row("방향 규약", r.get("orientation_contract") == REQUIRED_ORIENT,
        r.get("orientation_contract"), REQUIRED_ORIENT)
    row("회전 픽셀에 반영", r.get("rotation_baked") is True, r.get("rotation_baked"), "True")
    _fr = float(r["frames"]) if r.get("frames") else None
    # 하한은 필수다 — 60프레임(2초) 밑이면 접근·파지가 담길 수 없다
    over("프레임 수 하한", _fr, g["frames_min"], "{:.0f}")
    # 상한은 **권고**다. 길다는 건 난이도 경고지 폐기 사유가 아니다
    under("프레임 수 상한(권고)", _fr, g["frames_max"], "{:.0f}", soft=True)
    band("표본율 Hz", r.get("fps_median"), g["fps_lo"], g["fps_hi"])
    under("프레임 간격 최대/중앙", r.get("gap_ratio_max"), g["gap_ratio_max"])
    under("노출 변동계수", r.get("exposure_cv"), g["exposure_cv_max"])
    under("AF 이동폭", r.get("focus_span"), g["focus_span_max"])
    for tag, ko in (("accel", "가속도"), ("gyro", "자이로")):
        v = r.get(f"{tag}_hz")
        row(f"{ko} Hz", None if v is None else v >= g["imu_hz_min"],
            "없음 — 대조 불가" if v is None else f"{v:.1f}", f">= {g['imu_hz_min']}")
        mv = r.get(f"{tag}_margin_s")
        row(f"{ko} 프레임 구간 덮음", None if mv is None else mv >= g["imu_margin_s_min"],
            "없음 — 대조 불가" if mv is None else f"{mv:+.3f}s", f">= {g['imu_margin_s_min']}s")
    under("pts/센서 시각차 ms", r.get("pts_skew_ms_max"), g["pts_skew_ms_max"])
    # ⚠️ "없다"와 "모른다"를 가른다. 서비스 경로(gate_api)는 영상을 안 받으므로
    #    크기를 안 주면 **미판정**이지 불합격이 아니다. 폴더 경로에선 없으면 불합격이다.
    _vb = r.get("video_bytes", UNKNOWN)
    if _vb is UNKNOWN or _vb == "미지정":
        row("영상 파일", None, "크기 미지정 — 대조 불가", "있을 것")
    else:
        row("영상 파일", _vb is not None, f"{_vb} B" if _vb else "없음", "있을 것")

    # 판정은 **필수 항목만** 본다. 권고는 따로 센다 — 섞으면 길다는 이유로 전 편이 날아간다
    hard = [x for x in rows if not x["soft"]]
    soft = [x for x in rows if x["soft"]]
    p = sum(1 for x in hard if x["ok"] is True)
    f = sum(1 for x in hard if x["ok"] is False)
    u = sum(1 for x in hard if x["ok"] is None)
    return {"rows": rows, "passed": p, "failed": f, "unknown": u, "total": len(hard),
            "soft_total": len(soft), "soft_unmet": sum(1 for x in soft if x["ok"] is not True),
            "verdict": "PASS" if f == 0 and u == 0 else ("FAIL" if f else "INCOMPLETE")}


def _write_synth(root: Path, name: str, *, frames: int = 100, fps: float = 30.0,
                 outcome: str = "success", drop_at: int | None = None,
                 focus_moves: bool = False, imu: bool = True) -> Path:
    """Build a known-answer capture folder. 정답을 아는 촬영 폴더를 만든다."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    t0 = 759_643_000_000_000
    step = int(NS / fps)
    ts = []
    for i in range(frames):
        extra = step * 3 if (drop_at is not None and i == drop_at) else 0
        ts.append((ts[-1] if ts else t0) + (step + extra if ts else 0))
    (d / "manifest.json").write_text(json.dumps({
        "status": "recorded", "outcome": outcome, "image_orientation_contract": "upright_v1",
        "rotation_baked_into_pixels": True, "marker_black_square_size_verified": False,
        "training_ready": False, "width": 1920, "height": 1080, "device": "SM-S901N",
        "task": "grasp_lift_hold", "start_elapsed_ns": t0 - 300_000_000,
        "stop_elapsed_ns": ts[-1] + 300_000_000}), encoding="utf-8")
    with (d / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_number", "sensor_timestamp_ns", "exposure_ns", "iso", "focus_diopters"])
        for i, t in enumerate(ts):
            w.writerow([i, t, 8333700, 85, (0.01 * i) if focus_moves else 0.0])
    with (d / "encoded.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "pts_us", "size_bytes", "flags"])
        for i, t in enumerate(ts):
            w.writerow([i, t // 1000, 200000, 1 if i == 0 else 0])
    if imu:
        for fn in ("accelerometer.csv", "gyroscope.csv"):
            with (d / fn).open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_ns", "x_m_s2", "y_m_s2", "z_m_s2", "accuracy"])
                n = int((ts[-1] - ts[0] + 400_000_000) / NS * 200)
                for k in range(n):
                    w.writerow([ts[0] - 200_000_000 + int(k * NS / 200), 0.0, 0.0, 9.8, 3])
    (d / "video.mp4").write_bytes(b"\x00" * 1024)
    return d


def selftest() -> int:
    """Known-answer and discriminating rows. 정답 아는 행과 판별행."""
    import tempfile
    log, bad = [], 0

    def chk(name, cond, note=""):
        nonlocal bad
        log.append((name, bool(cond), note))
        if not cond:
            bad += 1

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        good = measure_episode(_write_synth(root, "good"))
        v = judge(good)
        chk("1 정상 촬영 -> PASS", v["verdict"] == "PASS",
            f"통과 {v['passed']}/{v['total']} · {v['verdict']}")
        chk("2 프레임 수·표본율 정답", good["frames"] == 100 and abs(good["fps_median"] - 30) < 0.1,
            f"{good['frames']}프레임 · {good['fps_median']:.2f}Hz")

        short = judge(measure_episode(_write_synth(root, "short", frames=40)))
        chk("3 40프레임 -> FAIL (판별행)", short["verdict"] == "FAIL", short["verdict"])
        _write_synth(root, "rec_long", frames=250)
        lng = judge(measure_episode(root / "rec_long"))
        chk("3b 250프레임 -> 권고만 미충족, 판정 PASS (정답 아는 행)",
            lng["verdict"] == "PASS" and lng["soft_unmet"] == 1,
            f"{lng['verdict']} · 권고 미충족 {lng['soft_unmet']}/{lng['soft_total']}")
        _write_synth(root, "rec_50", frames=50)
        shrt2 = judge(measure_episode(root / "rec_50"))
        _g = dict(good); _g.pop("video_bytes", None)        # 크기를 **안 알려준** 경우
        _ju = judge(_g)
        chk("3d 영상 크기 미지정 -> 미판정, PASS 아님 (판별행)",
            _ju["verdict"] == "INCOMPLETE" and _ju["unknown"] == 1,
            f"{_ju['verdict']} · 미판정 {_ju['unknown']}")
        _g2 = dict(good); _g2["video_bytes"] = None         # 영상이 **없는** 경우
        chk("3e 영상 없음 -> 불합격 (미판정과 갈린다)",
            judge(_g2)["verdict"] == "FAIL", judge(_g2)["verdict"])
        chk("3c 50프레임 -> 하한 필수 불합격 (판별행)",
            shrt2["verdict"] == "FAIL", shrt2["verdict"])

        drop = measure_episode(_write_synth(root, "drop", drop_at=50))
        chk("4 프레임 드롭 잡힌다 (판별행)", judge(drop)["verdict"] == "FAIL",
            f"간격 최대/중앙 {drop['gap_ratio_max']:.2f}")

        fail = judge(measure_episode(_write_synth(root, "failout", outcome="fail")))
        chk("5 실패 시연 -> FAIL (판별행)", fail["verdict"] == "FAIL", fail["verdict"])

        af = measure_episode(_write_synth(root, "af", focus_moves=True))
        chk("6 AF 이동 잡힌다 (판별행)", judge(af)["verdict"] == "FAIL",
            f"초점 이동폭 {af['focus_span']:.2f}")

        noimu = measure_episode(_write_synth(root, "noimu", imu=False))
        j = judge(noimu)
        chk("7 IMU 없음 -> INCOMPLETE (통과 아님)", j["verdict"] == "INCOMPLETE",
            f"미판정 {j['unknown']} · {j['verdict']}")

        empty = (root / "empty"); empty.mkdir()
        j2 = judge(measure_episode(empty))
        chk("8 빈 폴더 -> 통과 0", j2["passed"] == 0 and j2["verdict"] != "PASS",
            f"통과 {j2['passed']}/{j2['total']} · {j2['verdict']}")

        chk("9 모수 보고", v["total"] == 15, f"검사 항목 {v['total']}개")

    for nm, ok, note in log:
        print(f"  {'OK ' if ok else 'FAIL'}  {nm}" + (f"   {note}" if note else ""))
    print(f"\n자체검증 {len(log) - bad}/{len(log)}")
    return 1 if bad else 0


def main() -> None:
    """CLI entry point. 명령행 진입점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw", help="rec_* 들이 들어 있는 디렉터리")
    ap.add_argument("--episode", help="한 편만 검사")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 수치를 내지 않는다")
    if not a.raw:
        ap.error("--raw 가 필요하다")

    root = Path(a.raw).expanduser()
    eps = [root / a.episode] if a.episode else sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.startswith("rec_"))
    if not eps:
        raise SystemExit(f"!! {root} 안에 rec_* 가 없다. 경로가 맞는지 먼저 봐라")

    docs, tally = [], {"PASS": 0, "FAIL": 0, "INCOMPLETE": 0}
    for d in eps:
        r = measure_episode(d)
        j = judge(r)
        r["gates"] = j
        docs.append(r)
        tally[j["verdict"]] = tally.get(j["verdict"], 0) + 1
        mark = {"PASS": "통과", "FAIL": "불합격", "INCOMPLETE": "미판정"}[j["verdict"]]
        miss = [x for x in j["rows"] if x["ok"] is not True and not x["soft"]]
        smiss = [x for x in j["rows"] if x["ok"] is not True and x["soft"]]
        print(f"[{mark:^4}] {d.name:<34} 필수 {j['passed']}/{j['total']}"
              + (f" · 권고 미충족 {j['soft_unmet']}/{j['soft_total']}" if j["soft_unmet"] else "")
              + ("   " + " · ".join(f"{x['name']} {x['got']}" for x in miss) if miss else "")
              + ("   (권고) " + " · ".join(f"{x['name']} {x['got']}" for x in smiss) if smiss else ""))

    n = len(eps)
    print(f"\n통과 {tally['PASS']} · 불합격 {tally['FAIL']} · 미판정 {tally['INCOMPLETE']} / 전체 {n}")
    soft_hits = sum(1 for d0 in docs if d0["gates"]["soft_unmet"])
    if soft_hits:
        print(f"권고 미충족 {soft_hits} / {n} 편 — 폐기 사유가 아니다. 다음 촬영에서 줄여라")
    if tally["PASS"] < n:
        print("** 미판정은 통과가 아니다. 불합격 편은 다시 찍는다 — 게이트를 낮추지 않는다 **")
    # 부분합이 전체와 다르면 죽인다: 필수 한 항목이 전 편을 떨어뜨리면 게이트 쪽을 의심한다
    per = {}
    for d0 in docs:
        for x in d0["gates"]["rows"]:
            if x["ok"] is not True and not x["soft"]:
                per[x["name"]] = per.get(x["name"], 0) + 1
    if per and max(per.values()) == n:
        worst = [k for k, v in per.items() if v == n]
        print(f"\n** 필수 {worst} 가 전 {n} 편을 떨어뜨린다 — 데이터가 아니라 "
              f"게이트가 틀렸을 수 있다. 참조 배치에 먼저 걸어봐라 **")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"raw": str(root), "episodes": docs, "tally": tally, "total": n,
             "gates": GATES}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
