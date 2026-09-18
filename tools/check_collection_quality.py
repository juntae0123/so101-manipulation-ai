"""Screen a UMI collection batch before SLAM and conversion are spent on it.
SLAM·변환에 시간을 쓰기 전에 수집 배치를 전수 선별한다.

왜 (2026-09-18)
---------------
2026-09-18 수집분 92편이 들어왔다. 트랙 A 가 ORB-SLAM3 매핑과 v10 변환에 들어가기
전에 **불량 편을 걸러 보내는 것**이 이 도구의 목적이다.

앞선 사례가 근거다.
- 2026-09-17 수집분 2편: 턱 마커 **검출 0건** (ArUco 사전 27종 전수 확인).
  이런 편은 `gripper.csv` 가 빈 채로 파이프라인을 통과한다.
- BE 계약(`모델학습유의사항.md` 5.1): *"gripper.csv 가 헤더만 있고 0건이어도
  입력 자체를 거부하지 않는다."* → **에러 없이 학습이 나빠진다.**
- E2 측정에서 gap 채널 개선이 +54.9% 로 세 채널 중 가장 컸다. 빈 채널은 비싸다.

게이트 — 결과를 보기 전에 확정했다
----------------------------------
**확정 불합격** (근거가 명확한 것만)
    SYNC   frames.csv ↔ encoded.csv 타임스탬프 불일치 > 1 us
    SYNC   IMU 가 카메라 구간을 덮지 못함 (보간 불가 구간 발생)
    MARK   두 턱 마커 동시 검출 0 프레임 → gap 채널 생성 불가
    LEN    길이 < 3.0 초 → 파지·리프트·홀드가 들어갈 수 없다

**관찰만** (임계를 지금 박지 않는다. 분포를 본 적이 없다)
    마커 검출률 · 최장 연속 미검출 구간 · 프레임 드롭률 · 노출/ISO 변동

⚠️ 검출률 임계를 임의로 정하면 그건 계측이 아니라 취향이다. 분포를 먼저 내고
   트랙 A 와 같이 정한다.

⚠️ 이 도구는 **수집 품질**만 본다. SLAM 추적 성공 여부, 좌표계 정합, 스케일 배율은
   여기서 판정하지 않는다. 특히 스케일은 `check_marker_scale.py` 의 영역이고,
   그것도 **빈 그리퍼 캘리브레이션 클립**이 있어야 유효하다.

Usage
-----
  python check_collection_quality.py --selftest
  python check_collection_quality.py --root <수집 루트> --out quality.json
  python check_collection_quality.py --root <수집 루트> --stride 2 --limit 10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

MIN_SECONDS = 3.0
SYNC_TOL_US = 1


# ── 순수 계산 — 자체검증 대상 ─────────────────────────────────────────────

def sync_mismatch_us(frame_ns: list[int], pts_us: list[int]) -> tuple[int, int, int]:
    """Max |frames.csv ns/1000 - encoded.csv us| over paired rows.
    두 표의 짝지은 행에서 최대 타임스탬프 불일치. 반환 (최대불일치, 비교쌍, 전체행)."""
    n = min(len(frame_ns), len(pts_us))
    if n == 0:
        return (10 ** 9, 0, max(len(frame_ns), len(pts_us)))
    d = max(abs(frame_ns[i] // 1000 - pts_us[i]) for i in range(n))
    return (d, n, max(len(frame_ns), len(pts_us)))


def covers(outer: list[int], inner: list[int]) -> bool:
    """Does the IMU span cover the camera span? IMU 구간이 카메라 구간을 덮나."""
    if not outer or not inner:
        return False
    return outer[0] <= inner[0] and outer[-1] >= inner[-1]


def drop_rate(ts_ns: list[int], nominal_ns: int) -> tuple[int, int]:
    """Gaps longer than 1.5x nominal, EXCLUDING the first interval.
    공칭 간격의 1.5배를 넘는 간격 수 / 전체 간격. **첫 간격은 제외한다.**

    왜 첫 간격을 빼는가 (2026-09-18 실측 🟢)
    ----------------------------------------
    0918 수집분 92편 전부에서 index 0 간격이 정확히 공칭의 2배로 나왔다
    (66662656ns vs 공칭 33333333ns). 나머지 간격은 전부 33331355ns 로 평탄하다.
    카메라 워밍업 아티팩트이고 **전 편 공통이라 판별력이 0** 이다.
    이걸 세면 drops 가 92/92 전부 1 이 되어 "드롭 없음"과 "드롭 있음"이
    같은 출력으로 나온다. 첫 간격은 first_gap_ratio 로 따로 보고한다."""
    if len(ts_ns) < 3 or nominal_ns <= 0:
        return (0, 0)
    gaps = [ts_ns[i + 1] - ts_ns[i] for i in range(len(ts_ns) - 1)][1:]
    return (sum(1 for g in gaps if g > nominal_ns * 1.5), len(gaps))


def first_gap_ratio(ts_ns: list[int], nominal_ns: int) -> float:
    """First inter-frame gap as a multiple of nominal. 첫 간격이 공칭의 몇 배인가."""
    if len(ts_ns) < 2 or nominal_ns <= 0:
        return 0.0
    return (ts_ns[1] - ts_ns[0]) / nominal_ns


def longest_run(flags: list[bool], value: bool = False) -> int:
    """Longest consecutive run of `value`. 같은 값이 연속된 최장 길이."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f == value else 0
        best = max(best, cur)
    return best


def cv(xs: list[float]) -> float:
    """Coefficient of variation. 변동계수 = 표준편차 / 평균."""
    xs = [x for x in xs if math.isfinite(x)]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    if abs(m) < 1e-12:
        return 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var) / abs(m)


# ── 자체검증 — 정답을 아는 입력만 ─────────────────────────────────────────

def selftest() -> int:
    bad = 0

    d, n, tot = sync_mismatch_us([1_000_500, 2_000_499], [1000, 2000])
    ok = d == 0 and n == 2 and tot == 2
    print(f"[1] 동기화 일치 → 불일치 {d}us · 쌍 {n}/{tot}  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    d, _, _ = sync_mismatch_us([1_000_500], [1005])
    ok = d == 5
    print(f"[2] 5us 어긋남 감지 → {d}us  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    d, n, tot = sync_mismatch_us([], [1000, 2000])
    ok = d > SYNC_TOL_US and n == 0 and tot == 2
    print(f"[3] 빈 표 거부 → 불일치 {d} · 쌍 {n}/{tot}  ", end="")
    print("OK" if ok else "!! 실패 — 빈 입력을 통과시켰다"); bad += (not ok)

    ok = covers([0, 100], [10, 90]) and not covers([10, 90], [0, 100])
    print(f"[4] 구간 포함 판정  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    k, tot = drop_rate([0, 100, 200, 400, 500], 100)
    ok = k == 1 and tot == 3
    print(f"[5] 드롭 1개 감지(첫 간격 제외) → {k}/{tot}  기대 1/3  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    k, tot = drop_rate([0, 200, 300, 400, 500], 100)
    ok = k == 0 and tot == 3
    print(f"[5b] 첫 간격 2배는 드롭 아님 → {k}/{tot}  기대 0/3  ", end="")
    print("OK" if ok else "!! 실패 — 워밍업을 드롭으로 센다"); bad += (not ok)

    r = first_gap_ratio([0, 200, 300], 100)
    ok = abs(r - 2.0) < 1e-9
    print(f"[5c] 첫 간격 배율 → {r:.2f}  기대 2.00  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    r = longest_run([True, False, False, False, True, False], False)
    ok = r == 3
    print(f"[6] 최장 미검출 연속 → {r}  기대 3  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    ok = abs(cv([10, 10, 10])) < 1e-12 and cv([1, 2, 3]) > 0.4
    print(f"[7] 변동계수 상수 0 · 변동 감지  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


# ── 편별 검사 ────────────────────────────────────────────────────────────

def read_col(p: Path, col: str) -> list[int]:
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return [int(float(r[col])) for r in csv.DictReader(f) if r.get(col)]


def detect_markers(video: Path, ids: tuple[int, int], stride: int) -> tuple[list[bool], int]:
    """Per-sampled-frame: both jaw markers visible? 샘플 프레임마다 두 마커가 다 보이나."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return ([], 0)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        run = det.detectMarkers
    else:
        pr = cv2.aruco.DetectorParameters_create()
        run = lambda g: cv2.aruco.detectMarkers(g, d, parameters=pr)   # noqa: E731
    flags, i, read = [], 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        read += 1
        if i % stride:
            i += 1; continue
        i += 1
        _, mid, _ = run(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        s = set(int(x) for x in mid.flatten()) if mid is not None else set()
        flags.append(ids[0] in s and ids[1] in s)
    cap.release()
    return (flags, read)


def inspect(d: Path, stride: int, skip_video: bool) -> dict:
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    fts = read_col(d / "frames.csv", "sensor_timestamp_ns")
    ets = read_col(d / "encoded.csv", "pts_us")
    ats = read_col(d / "accelerometer.csv", "timestamp_ns")
    gts = read_col(d / "gyroscope.csv", "timestamp_ns")
    exp = read_col(d / "frames.csv", "exposure_ns")
    iso = read_col(d / "frames.csv", "iso")
    nominal = read_col(d / "frames.csv", "frame_duration_ns")

    mis, pairs, rows = sync_mismatch_us(fts, ets)
    imu_ok = covers(ats, fts) and covers(gts, fts)
    nom0 = nominal[0] if nominal else 0
    dk, dtot = drop_rate(fts, nom0)
    fgr = first_gap_ratio(fts, nom0)
    dur = (m.get("stop_elapsed_ns", 0) - m.get("start_elapsed_ns", 0)) / 1e9

    r = {"session": d.name, "group": d.parent.name,
         "duration_s": round(dur, 2), "task": m.get("task"), "outcome": m.get("outcome"),
         "training_ready": m.get("training_ready"), "calibration_id": m.get("calibration_id"),
         "frames": len(fts), "sync_mismatch_us": mis, "sync_pairs": pairs, "sync_rows": rows,
         "imu_covers": imu_ok, "imu_accel": len(ats), "imu_gyro": len(gts),
         "drops": dk, "gaps": dtot, "first_gap_ratio": round(fgr, 3),
         "exposure_cv": round(cv([float(x) for x in exp]), 4),
         "iso_cv": round(cv([float(x) for x in iso]), 4),
         "marker_frames": None, "marker_hits": None, "marker_rate": None,
         "marker_longest_miss": None, "stride": stride}

    if not skip_video:
        ids = (int(m.get("gripper_left_id", 0)), int(m.get("gripper_right_id", 1)))
        flags, read = detect_markers(d / "video.mp4", ids, stride)
        r["marker_frames"] = len(flags)
        r["video_frames_read"] = read
        r["marker_hits"] = sum(flags)
        r["marker_rate"] = round(sum(flags) / len(flags), 4) if flags else 0.0
        r["marker_longest_miss"] = longest_run(flags, False)

    fail = []
    if mis > SYNC_TOL_US:
        fail.append(f"SYNC 불일치 {mis}us")
    if not imu_ok:
        fail.append("SYNC IMU 미포함")
    if dur < MIN_SECONDS:
        fail.append(f"LEN {dur:.2f}s < {MIN_SECONDS}")
    if r["marker_hits"] == 0 and not skip_video:
        fail.append("MARK 검출 0건")
    r["fail"] = fail
    r["pass"] = not fail
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", help="rec_* 디렉터리들을 담은 루트 (하위 폴더 포함)")
    ap.add_argument("--stride", type=int, default=1, help="마커 검출 프레임 간격")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-video", action="store_true", help="마커 검출 생략 (빠름)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.root:
        raise SystemExit("!! --root 가 필요하다")
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 판정하지 않는다")
    print()

    root = Path(a.root).expanduser()
    dirs = sorted(p for p in root.rglob("rec_*") if p.is_dir() and (p / "manifest.json").exists())
    total = len(dirs)
    if a.limit:
        dirs = dirs[:a.limit]
    print(f"대상 {len(dirs)} / 전체 {total} 편 · stride {a.stride}"
          f"{' · 영상 생략' if a.skip_video else ''}\n")

    rows = []
    for i, d in enumerate(dirs, 1):
        try:
            r = inspect(d, a.stride, a.skip_video)
        except Exception as exc:                                   # noqa: BLE001
            r = {"session": d.name, "group": d.parent.name, "pass": False,
                 "fail": [f"ERROR {type(exc).__name__}: {exc}"]}
        rows.append(r)
        mark = "OK " if r.get("pass") else "!! "
        mr = r.get("marker_rate")
        print(f"  [{i:3d}/{len(dirs)}] {mark}{r['session'][:26]:28s} "
              f"{r.get('duration_s','?'):>6}s  "
              f"마커 {('%.0f%%' % (mr*100)) if mr is not None else '  -':>5s}  "
              f"{' · '.join(r.get('fail', [])) or ''}", flush=True)

    ok = [r for r in rows if r.get("pass")]
    ng = [r for r in rows if not r.get("pass")]
    print(f"\n{'='*66}")
    print(f"합격 {len(ok)} / 검사 {len(rows)} / 전체 {total}")
    if ng:
        print(f"\n불합격 {len(ng)}편")
        for r in ng:
            print(f"  {r['group']}/{r['session']}  {' · '.join(r['fail'])}")

    def dist(key: str, fmt: str = "{:.3f}") -> None:
        v = sorted(r[key] for r in rows if r.get(key) is not None)
        if not v:
            print(f"  {key:22s} 값 없음"); return
        q = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1)))]      # noqa: E731
        print(f"  {key:22s} 최소 {fmt.format(v[0])} · 25% {fmt.format(q(.25))} · "
              f"중앙 {fmt.format(q(.5))} · 75% {fmt.format(q(.75))} · 최대 {fmt.format(v[-1])}"
              f"   n={len(v)}/{len(rows)}")

    print(f"\n분포 — 임계를 박지 않은 관찰 지표")
    dist("marker_rate"); dist("marker_longest_miss", "{:.0f}")
    dist("duration_s", "{:.2f}"); dist("iso_cv"); dist("exposure_cv")
    dist("drops", "{:.0f}"); dist("first_gap_ratio", "{:.2f}")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"root": str(root), "checked": len(rows), "total": total,
             "stride": a.stride, "skip_video": a.skip_video,
             "gates": {"min_seconds": MIN_SECONDS, "sync_tol_us": SYNC_TOL_US,
                       "note": "마커 검출률 임계는 의도적으로 두지 않았다. 분포를 보고 정한다"},
             "pass": [r["session"] for r in ok], "fail": ng, "rows": rows},
            indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
