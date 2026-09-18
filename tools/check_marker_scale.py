"""End-to-end scale check for S22 UMI recordings using the gripper jaw markers.
S22 수집분의 복원 스케일이 mm 로 맞는지 그리퍼 턱 마커로 검정한다.

무엇을 재나 — "마커 크기"가 아니라 "복원 스케일"이다
------------------------------------------------------
`manifest.json` 이 스케일 근거를 이렇게 적어 놓았다.

    "marker_side_mm_user_reported": 16
    "marker_black_square_size_verified": false      ← 미검증

이 16mm 가 틀리면 복원된 모든 거리가 같은 비율로 틀린다. 그런데 **마커 크기만 재도
소용없다.** 초점거리 오차도 똑같이 곱셈으로 들어오기 때문이다.

    복원거리 = 참거리 × (선언 마커크기 / 참 마커크기) × (가정 초점 / 참 초점)

우리를 망치는 건 이 **곱해진 최종 배율 하나**다. 그래서 그것만 잰다.

정답을 아는 행 — 리그에 이미 있다
----------------------------------
마커 0·1 이 그리퍼 양 턱에 붙어 있다 (`gripper_left_id` / `gripper_right_id`).
그리고 그리퍼 전행정은 **90.0 mm** 로 확정돼 있다.

    URDF        gripper prismatic [0, 0.09] m · 좌우 각 0.045
    MODELING_SPEC  w=0/0.04/0.09 → 접촉면 간격 0/40/90 mm (메시 검증)
    HW 실측      약 90mm / 0mm (2026-09-18)

따라서 **완전개방 프레임과 완전폐쇄 프레임의 마커 중심간 거리 차이 = 90.0 mm** 여야 한다.
별도 출력물도, 자도 필요 없다.

    scale_error = 측정된 전행정 / 90.0

두 가지 추정기를 같이 돌린다
-----------------------------
**A. 비율법 (내부파라미터 불필요)** — 마커 변의 픽셀 길이를 자로 쓴다.

    거리_mm = (중심간 픽셀거리 / 마커 변 픽셀길이) × 선언_마커_mm

    초점거리 오차에 **불변**이다. 두 마커가 같은 평면에 있고 정면에 가까울 때 유효.

**B. PnP 법 (내부파라미터 필요)** — solvePnP 로 3D 자세를 푼다.

    내부파라미터가 없으면 manifest 의 FOV 로 근사한다.
    ⚠️ 울트라와이드(FOV 104도)라 왜곡이 크다. **근사값임을 반드시 병기한다.**

**두 값이 다르면 그 차이가 원근·왜곡의 크기다.** 하나만 재면 그걸 못 본다.

⚠️ 한계 — 먼저 말한다
- 마커 중심간 거리 ≠ 턱 접촉면 간격. **차이(전행정)만 의미가 있고 절대값은 아니다**
- 완전개방·완전폐쇄 프레임이 영상에 실제로 있어야 한다. 없으면 전행정이 과소 추정된다
- 마커가 턱에 대해 기울어 붙어 있으면 비율법이 그만큼 틀린다
- 이 검정은 **스케일**만 본다. 회전 규약·시간정렬은 별개다

Usage
-----
  python check_marker_scale.py --selftest
  python check_marker_scale.py --recording ~/s22_latest/rec_1789627458118_4d66022a
  python check_marker_scale.py --recording <dir> --expect-travel-mm 90.0 --save-plot out.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

EXPECT_TRAVEL_MM = 90.0          # 그리퍼 전행정. URDF·MODELING_SPEC·HW 실측 3중 확인


# ── 기하 ─────────────────────────────────────────────────────────────────

def marker_side_px(corners: np.ndarray) -> float:
    """Mean edge length of a 4-corner marker, in pixels. 마커 네 변의 평균 픽셀 길이."""
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return float(np.mean([np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]))


def marker_center_px(corners: np.ndarray) -> np.ndarray:
    """Centroid of the 4 corners. 네 코너의 무게중심."""
    return np.asarray(corners, dtype=np.float64).reshape(4, 2).mean(axis=0)


def ratio_distance_mm(c0: np.ndarray, c1: np.ndarray, side_mm: float) -> float:
    """Center distance in mm using the markers themselves as the ruler.
    마커 변을 자로 써서 중심간 거리를 mm 로. 초점거리 오차에 불변."""
    px = float(np.linalg.norm(marker_center_px(c0) - marker_center_px(c1)))
    scale = 0.5 * (marker_side_px(c0) + marker_side_px(c1))    # 두 마커 평균
    if scale <= 1e-9:
        return float("nan")
    return px / scale * side_mm


def intrinsics_from_fov(width: int, height: int, hfov_deg: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole intrinsics approximated from horizontal FOV. FOV 로부터 근사 내부파라미터.
    ⚠️ 왜곡을 무시한다. 울트라와이드에서는 근사다."""
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)
    return K, np.zeros((5, 1), dtype=np.float64)


# ── 자체 검증 — 정답을 아는 합성 입력만 쓴다 ─────────────────────────────

def _square(cx: float, cy: float, side: float, rot_deg: float = 0.0) -> np.ndarray:
    h = side / 2.0
    pts = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64)
    t = math.radians(rot_deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return (pts @ R.T) + np.array([cx, cy])


def selftest() -> int:
    bad = 0

    # [1] 변 100px · 중심간 500px · 선언 16mm → 500/100×16 = 80.0mm
    a = _square(0, 0, 100); b = _square(500, 0, 100)
    got = ratio_distance_mm(a, b, 16.0)
    ok = abs(got - 80.0) < 1e-6
    print(f"[1] 비율법 기본      {got:.6f} mm  기대 80.000000  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    # [2] 전체를 2배 확대해도 같은 값 (스케일 불변)
    a2 = _square(0, 0, 200); b2 = _square(1000, 0, 200)
    got2 = ratio_distance_mm(a2, b2, 16.0)
    ok = abs(got2 - got) < 1e-6
    print(f"[2] 2배 확대 불변    {got2:.6f} mm  기대 {got:.6f}  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    # [3] 마커를 37도 돌려도 변 길이가 같아야 한다 (회전 불변)
    ar = _square(0, 0, 100, 37.0)
    ok = abs(marker_side_px(ar) - 100.0) < 1e-6
    print(f"[3] 회전 불변 변길이  {marker_side_px(ar):.6f} px  기대 100.000000  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    # [4] 고의 손상 — 한 마커만 절반 크기로. 값이 반드시 달라져야 한다
    bad_b = _square(500, 0, 50)
    got4 = ratio_distance_mm(a, bad_b, 16.0)
    ok = abs(got4 - 80.0) > 1.0
    print(f"[4] 손상 입력 감지    {got4:.3f} mm (80 과 달라야 함)  ", end="")
    print("OK" if ok else "!! 실패 — 조용히 통과했다"); bad += (not ok)

    # [5] FOV → 초점거리. FOV 90도, 폭 1920 이면 f = 960
    K, _ = intrinsics_from_fov(1920, 1080, 90.0)
    ok = abs(K[0, 0] - 960.0) < 1e-6
    print(f"[5] FOV 90도 → f     {K[0,0]:.6f} px  기대 960.000000  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    # [6] 전행정 환산 — 개방 95mm, 폐쇄 5mm 면 전행정 90mm, 배율 1.000
    travel = 95.0 - 5.0
    ok = abs(travel / EXPECT_TRAVEL_MM - 1.0) < 1e-9
    print(f"[6] 전행정 배율 계산  {travel/EXPECT_TRAVEL_MM:.6f}  기대 1.000000  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


# ── 본체 ─────────────────────────────────────────────────────────────────

def detect(video: Path, ids_wanted: tuple[int, int], max_frames: int
           ) -> tuple[list[dict], int, int]:
    """Detect the two jaw markers per frame. 프레임마다 턱 마커 두 개를 검출한다.
    반환: (검출된 프레임 목록, 읽은 프레임 수, 전체 프레임 수)"""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"!! 영상을 못 연다: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):                    # OpenCV 4.7+
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        run = lambda g: det.detectMarkers(g)                    # noqa: E731
    else:                                                      # 구버전
        p = cv2.aruco.DetectorParameters_create()
        run = lambda g: cv2.aruco.detectMarkers(g, d, parameters=p)   # noqa: E731

    out, read = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and read >= max_frames):
            break
        read += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = run(gray)
        if ids is None:
            continue
        flat = {int(i): c for i, c in zip(ids.flatten(), corners)}
        if ids_wanted[0] in flat and ids_wanted[1] in flat:
            out.append({"frame": read - 1,
                        "c0": np.asarray(flat[ids_wanted[0]]).reshape(4, 2),
                        "c1": np.asarray(flat[ids_wanted[1]]).reshape(4, 2)})
    cap.release()
    return out, read, total


def run(a) -> None:
    rec = Path(a.recording).expanduser()
    man = json.loads((rec / "manifest.json").read_text(encoding="utf-8"))
    cam = man["camera"]
    side_mm = a.marker_mm if a.marker_mm else float(man["marker_side_mm_user_reported"])
    ids = (int(man.get("gripper_left_id", 0)), int(man.get("gripper_right_id", 1)))
    verified = man.get("marker_black_square_size_verified", False)

    print(f"수집분 {rec.name}")
    print(f"  선언 마커 변 {side_mm} mm · 검증됨 {verified}")
    print(f"  턱 마커 id {ids} · {man['width']}x{man['height']} · FOV {cam['horizontal_fov_estimate']:.2f}도")
    if not verified:
        print("  ⚠️ manifest 가 마커 크기를 미검증으로 표시하고 있다. 그래서 이 검정을 한다")

    hits, read, total = detect(rec / "video.mp4", ids, a.max_frames)
    print(f"\n검출 {len(hits)} / 읽은 프레임 {read} / 영상 전체 {total}")
    if len(hits) < 5:
        raise SystemExit("!! 두 마커가 동시에 보이는 프레임이 5개 미만이다. "
                         "판정하지 않는다 (개방·폐쇄 양 끝이 필요하다)")

    # A. 비율법
    ratio = np.array([ratio_distance_mm(h["c0"], h["c1"], side_mm) for h in hits])
    ratio = ratio[np.isfinite(ratio)]

    # B. PnP 법 (근사 내부파라미터)
    pnp = None
    try:
        import cv2
        K, dist = intrinsics_from_fov(int(man["width"]), int(man["height"]),
                                      float(cam["horizontal_fov_estimate"]))
        obj = np.array([[-side_mm / 2, side_mm / 2, 0], [side_mm / 2, side_mm / 2, 0],
                        [side_mm / 2, -side_mm / 2, 0], [-side_mm / 2, -side_mm / 2, 0]],
                       dtype=np.float64)
        vals = []
        for h in hits:
            ts = []
            for key in ("c0", "c1"):
                ok, _, t = cv2.solvePnP(obj, h[key].astype(np.float64), K, dist,
                                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    ts = []; break
                ts.append(t.reshape(3))
            if len(ts) == 2:
                vals.append(float(np.linalg.norm(ts[0] - ts[1])))
        pnp = np.array(vals) if vals else None
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (PnP 생략: {type(exc).__name__}: {exc})")

    def report(name: str, v: np.ndarray, note: str = "") -> dict:
        lo, hi = float(v.min()), float(v.max())
        travel = hi - lo
        k = travel / a.expect_travel_mm
        print(f"\n[{name}] {note}")
        print(f"  중심간 거리  최소 {lo:7.2f} mm   최대 {hi:7.2f} mm   중앙 {float(np.median(v)):7.2f} mm")
        print(f"  전행정       {travel:7.2f} mm   기대 {a.expect_travel_mm:.2f} mm")
        print(f"  **스케일 배율 {k:.4f}**  (1.0 이 정확. {(k-1)*100:+.1f}%)")
        return {"min_mm": lo, "max_mm": hi, "median_mm": float(np.median(v)),
                "travel_mm": travel, "scale_factor": k, "n": int(v.size)}

    rep = {"recording": rec.name, "marker_side_mm_declared": side_mm,
           "marker_verified": verified, "ids": list(ids),
           "frames_detected": len(hits), "frames_read": read, "frames_total": total,
           "expect_travel_mm": a.expect_travel_mm,
           "expect_travel_source": "URDF gripper [0,0.09] · MODELING_SPEC 메시검증 · HW 실측 2026-09-18"}
    rep["ratio_method"] = report("A 비율법", ratio, "내부파라미터 불필요 · 초점거리 오차에 불변")
    if pnp is not None and pnp.size:
        rep["pnp_method"] = report("B PnP 법", pnp,
                                   "⚠️ FOV 로 근사한 내부파라미터 · 울트라와이드 왜곡 미보정")
        d = abs(rep["ratio_method"]["scale_factor"] - rep["pnp_method"]["scale_factor"])
        print(f"\n두 추정기 배율 차이 {d:.4f} — 이 값이 원근·왜곡의 크기다")
        rep["method_gap"] = d

    k = rep["ratio_method"]["scale_factor"]
    print("\n" + "=" * 62)
    if abs(k - 1.0) <= 0.03:
        print(f"판정: 스케일 오차 {abs(k-1)*100:.1f}% — 3% 이내. 16mm 가정을 유지한다")
    else:
        implied = side_mm / k
        print(f"판정: 스케일 오차 {abs(k-1)*100:.1f}% — 3% 초과. **미검증 항목이 실제로 틀렸다**")
        print(f"      역산한 참 마커 변 ≈ {implied:.2f} mm (초점 오차가 없다는 가정 하에)")
        print(f"      ⚠️ 초점거리 오차와 분리되지 않는다. 자로 마커를 직접 재서 대조하라")
    print("⚠️ 이 검정은 스케일만 본다. 회전 규약·시간정렬은 별개다")

    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"→ {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--recording", help="manifest.json · video.mp4 가 있는 디렉터리")
    ap.add_argument("--marker-mm", type=float, default=None,
                    help="선언 마커 변. 생략하면 manifest 값을 쓴다")
    ap.add_argument("--expect-travel-mm", type=float, default=EXPECT_TRAVEL_MM)
    ap.add_argument("--max-frames", type=int, default=0, help="0이면 전부")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.recording:
        raise SystemExit("!! --recording 이 필요하다")
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 측정하지 않는다")
    print()
    run(a)


if __name__ == "__main__":
    main()
