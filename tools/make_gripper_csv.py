"""Derive gripper.csv from UMI frames by detecting the two magenta markers.
UMI 프레임에서 마젠타 마커 2개를 검출해 gripper.csv 를 만든다.

왜 있는가 / Why this exists.

번들 71편에 `gripper.csv` 가 없다. `episode.json` 이 스스로 그 값을
`"postprocess: tools/gripper_gap.py"` 라고 적지만 그 후처리가 안 돌았다.
`gap_m` 이 없으면 그리퍼 개폐 라벨을 만들 수 없고, 2026-09-10 실측 🟢 으로
정책의 병목이 정확히 "언제 닫는가" 임이 확인됐다. 그래서 71편이 전부 막힌다.

⚠️ **이것은 트랙 A(황도경) 영역의 재구현이다.** 원본 `tools/gripper_gap.py` 가
   오면 **그쪽이 정본이다.** 여기 결과는 `gripper_gap_source=reimpl` 로 표시해
   원본 산출물과 섞이지 않게 한다.

교정 상수의 출처 — 섞지 않는다:
  🔵 구두 (황도경, 2026-09-12): 마커 지름 15mm · 마커 중심거리 37.6mm 에서 gap 0mm ·
     134mm 에서 gap 70mm 의 선형식
  🟢 실측 (이 저장소, 2026-09-12): `SCALE_CORRECTION` 은 도경이 낸
     `rec_20260911_150920` 76프레임에 최소제곱 적합한 값이다.
     **아래 MASK_* 임계값에 묶여 있다 — 임계값을 바꾸면 이 계수는 무효다.**

fixture 로 검정한 결과 🟢 (rec_20260911_150920, n=76):
  상관 0.999543 · 중앙오차 0.348mm · 최대 1.465mm · 폐쇄 전이 프레임 40 일치
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- 교정 상수 🔵 (황도경 구두, 2026-09-12) ------------------------------------
MARKER_DIAMETER_MM = 15.0
DIST_AT_GAP_MIN_MM = 37.6        # 마커 중심거리 37.6mm -> gap 0mm
DIST_AT_GAP_MAX_MM = 134.0       # 마커 중심거리 134mm  -> gap 70mm
GAP_AT_DIST_MAX_MM = 70.0

# --- 검출 임계값 + 그에 묶인 보정계수 🟢 ---------------------------------------
# 형광 마젠타: R·B 가 높고 G 가 낮다. 값을 바꾸면 SCALE_CORRECTION 을 다시 적합해야 한다.
MASK_R_MIN, MASK_B_MIN, MASK_RG_MIN, MASK_BG_MIN = 110, 70, 60, 25
MIN_BLOB_PIXELS = 200
SCALE_CORRECTION = 0.97859       # 이 마스크가 마커를 +2.19% 크게 잡는 것을 상쇄한다

# --- 유효성 검사 -------------------------------------------------------------
# 첫 판(2026-09-12)에는 이게 없었고, 71편 전체에서 gap 이 -0.6 ~ 376.3mm 로 나왔다.
# 물리적 상한이 90mm 인데 376mm 를 **조용히** 내놓았다 -- 2-means 가 마커가 아닌
# 마젠타 영역(혹은 마커 하나가 둘로 쪼개진 것)을 잡은 것이다.
# 계측기가 스스로 고장을 알리지 않는 전형이라, 검사를 코드에 넣는다.
MAX_SIZE_RATIO = 1.6             # 같은 마커 둘이다. 투영 면적이 이보다 벌어지면 오검출
GAP_VALID_MM = (0.0, 90.0)       # umi/raw.py UMI_GAP_RANGE_M = (0.0, 0.090)
MAX_FILL_SLACK = 2.2             # blob 넓이 대비 경계상자 넓이. 원이면 약 4/pi=1.27

# --- fixture 게이트 (결과 보기 전에 고정) ---------------------------------------
FIXTURE_MEDIAN_MM = 0.5
FIXTURE_MAX_MM = 3.0

# SPEC 의 status. D=양쪽 직접검출 · M=중심선 대칭 추정 · T=템플릿 · X=실패.
# 재구현은 D 와 X 만 낸다 — 안 하는 추정을 했다고 적지 않는다.
STATUS_DETECTED = "D"
STATUS_FAILED = "X"


@dataclass
class FrameResult:
    """One frame's detection. gap_mm is None when detection failed.
    프레임 하나의 검출 결과. 실패면 gap_mm 이 None 이다."""

    index: int
    gap_mm: float | None
    dist_px: float | None
    diameter_px: float | None
    status: str


def detect(path: Path, draft: int = 1) -> FrameResult:
    """Find the two markers and convert their centre distance to a finger gap.
    마커 둘을 찾아 중심거리를 손가락 간격으로 바꾼다.

    2-means on x is enough: the two markers sit on opposite jaws and are the only
    large magenta regions in frame (실측: blob 2개가 각 ~12,000px, 다음이 296px).
    x 축 2-means 로 충분하다. 마커 둘은 반대쪽 턱에 붙어 있고 화면에서 유일하게 큰
    마젠타 영역이다. 연결요소 라이브러리를 끌어오지 않는 이유이기도 하다 (scipy 불필요).
    """
    idx = int(path.stem)
    img = Image.open(path)
    if draft > 1:
        # JPEG 을 1/draft 로 **디코드 단계에서** 줄인다. 1920x1080 전체를 푸는 것보다
        # 훨씬 싸다. 거리와 지름이 같은 비율로 줄어드므로 mm/px 환산은 불변이고,
        # SCALE_CORRECTION 도 그대로다 -- 단 마스크가 미세하게 달라지므로
        # **이 설정으로 fixture 를 다시 통과해야 쓴다.**
        img.draft("RGB", (img.size[0] // draft, img.size[1] // draft))
    im = np.asarray(img.convert("RGB"), dtype=np.int16)
    min_blob = max(20, MIN_BLOB_PIXELS // (draft * draft))
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mask = ((r > MASK_R_MIN) & (b > MASK_B_MIN)
            & (r - g > MASK_RG_MIN) & (b - g > MASK_BG_MIN))
    ys, xs = np.nonzero(mask)
    if xs.size < 2 * min_blob:
        return FrameResult(idx, None, None, None, STATUS_FAILED)

    centres = np.array([xs.min(), xs.max()], dtype=float)
    assign = np.zeros(xs.shape, dtype=np.int64)
    for _ in range(50):
        assign = np.abs(xs[:, None] - centres[None, :]).argmin(1)
        moved = np.array([xs[assign == k].mean() if (assign == k).any() else centres[k]
                          for k in (0, 1)])
        if np.allclose(moved, centres):
            break
        centres = moved

    left, right = assign == 0, assign == 1
    if left.sum() < min_blob or right.sum() < min_blob:
        return FrameResult(idx, None, None, None, STATUS_FAILED)

    # 2-means 는 마스크의 **모든** 픽셀을 둘 중 하나에 배정한다 -- 멀리 떨어진 잡음
    # 픽셀도 포함된다. 그대로 두면 경계상자 검사가 무의미해지고 중심도 끌려간다.
    # 각 덩이에서 중심으로부터 중앙거리의 2.5배 밖을 떨어낸다.
    def _trim(sel: np.ndarray) -> np.ndarray:
        cx, cy = xs[sel].mean(), ys[sel].mean()
        d = np.hypot(xs[sel] - cx, ys[sel] - cy)
        keep = d <= 2.5 * max(float(np.median(d)), 1.0)
        out = sel.copy()
        out[np.nonzero(sel)[0][~keep]] = False
        return out

    left, right = _trim(left), _trim(right)
    if left.sum() < min_blob or right.sum() < min_blob:
        return FrameResult(idx, None, None, None, STATUS_FAILED)

    n0, n1 = int(left.sum()), int(right.sum())
    if max(n0, n1) / min(n0, n1) > MAX_SIZE_RATIO:
        return FrameResult(idx, None, None, None, STATUS_FAILED)

    # 두 덩이가 각각 원에 가까운가. 마커 하나가 둘로 쪼개졌거나 마커 아닌 것이
    # 섞이면 경계상자가 넓이에 비해 커진다.
    for sel in (left, right):
        bw = xs[sel].max() - xs[sel].min() + 1
        bh = ys[sel].max() - ys[sel].min() + 1
        if bw * bh > MAX_FILL_SLACK * int(sel.sum()) * 4.0 / np.pi:
            return FrameResult(idx, None, None, None, STATUS_FAILED)

    cx0, cy0 = xs[left].mean(), ys[left].mean()
    cx1, cy1 = xs[right].mean(), ys[right].mean()
    dist_px = float(np.hypot(cx1 - cx0, cy1 - cy0))

    # 원 가정 지름. 프레임마다 다시 잰다 -- 에피소드마다 카메라-그리퍼 거리가 달라서
    # 한 편에서 적합한 상수 mm/px 는 다른 편에서 틀어진다.
    dia_px = float(np.mean([2.0 * np.sqrt(left.sum() / np.pi),
                            2.0 * np.sqrt(right.sum() / np.pi)]))
    mm_per_px = (MARKER_DIAMETER_MM / dia_px) * SCALE_CORRECTION
    dist_mm = dist_px * mm_per_px

    span_mm = DIST_AT_GAP_MAX_MM - DIST_AT_GAP_MIN_MM
    gap_mm = (dist_mm - DIST_AT_GAP_MIN_MM) / span_mm * GAP_AT_DIST_MAX_MM
    if not (GAP_VALID_MM[0] <= gap_mm <= GAP_VALID_MM[1]):
        # 물리적으로 불가능한 값은 채우지 않는다. 채우면 학습이 그것을 배운다.
        return FrameResult(idx, None, dist_px, dia_px, STATUS_FAILED)
    return FrameResult(idx, gap_mm, dist_px, dia_px, STATUS_DETECTED)


def run_episode(ep: Path, draft: int = 1) -> list[FrameResult]:
    frames = sorted((ep / "frames").glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"프레임이 없다: {ep}")
    return [detect(f, draft) for f in frames]


def write_csv(ep: Path, results: list[FrameResult], out_name: str) -> Path:
    """gap_m 은 미터다 (SPEC). 실패 프레임은 **빈 칸**으로 둔다 -- 0 으로 채우면
    '완전히 닫혔다'가 되어 학습이 그것을 배운다 (track_a/convert/arcore.py 의 규칙)."""
    path = ep / out_name
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_index", "gap_m", "status"])
        for r in results:
            w.writerow([r.index,
                        "" if r.gap_mm is None else f"{r.gap_mm / 1000.0:.9f}",
                        r.status])
    return path


def read_reference(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("gap_m"):
                out[int(row["frame_index"])] = float(row["gap_m"]) * 1000.0
    return out


def check_fixture(results: list[FrameResult], ref: dict[int, float]) -> tuple[bool, str]:
    """Refuse to trust our numbers unless they reproduce the reference file.
    기준 파일을 재현하지 못하면 우리 수치를 믿지 않는다."""
    pairs = [(r.gap_mm, ref[r.index]) for r in results
             if r.gap_mm is not None and r.index in ref]
    if len(pairs) < 0.9 * len(ref):
        return False, f"겹치는 프레임이 {len(pairs)}/{len(ref)} 뿐이다"
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    err = np.abs(a - b)
    med, mx = float(np.median(err)), float(err.max())
    corr = float(np.corrcoef(a, b)[0, 1])

    th = (b.max() + b.min()) / 2.0
    t_ref = int(np.argmax(b < th))
    t_our = int(np.argmax(a < th))

    ok = med <= FIXTURE_MEDIAN_MM and mx <= FIXTURE_MAX_MM and t_ref == t_our
    msg = (f"중앙 {med:.3f}mm (<= {FIXTURE_MEDIAN_MM}) · 최대 {mx:.3f}mm "
           f"(<= {FIXTURE_MAX_MM}) · 상관 {corr:.6f} · "
           f"폐쇄 전이 기준 {t_ref} / 우리 {t_our}")
    return ok, msg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-root", type=Path, required=True,
                    help="rec_* 폴더들이 있는 상위 디렉터리")
    ap.add_argument("--fixture", type=Path, default=None,
                    help="기준 gripper.csv. 주면 --fixture-episode 로 먼저 검정하고, "
                         "게이트에 걸리면 **아무것도 쓰지 않는다**")
    ap.add_argument("--fixture-episode", type=str, default=None,
                    help="기준 파일이 나온 rec_* 이름")
    ap.add_argument("--out-name", type=str, default="gripper_reimpl.csv",
                    help="원본 gripper.csv 와 섞이지 않게 기본 이름을 다르게 둔다")
    ap.add_argument("--only", type=str, default=None, help="이 rec_* 하나만 처리")
    ap.add_argument("--draft", type=int, default=1,
                    help="JPEG 을 1/N 로 디코드해 가속한다. 바꾸면 fixture 를 다시 통과해야 한다")
    ap.add_argument("--skip-existing", action="store_true",
                    help="이미 out-name 이 있는 에피소드는 건너뛴다 (중단 후 재개)")
    ap.add_argument("--dry-run", action="store_true", help="fixture 검정만 하고 쓰지 않는다")
    args = ap.parse_args()

    root = args.bundle_root
    eps = sorted(p for p in root.glob("rec_*") if (p / "frames").is_dir())
    if args.only:
        eps = [p for p in eps if p.name == args.only]
    if not eps:
        print(f"✗ 에피소드가 없다: {root}")
        return 2
    print(f"에피소드 {len(eps)}편")

    if args.fixture is not None:
        if args.fixture_episode is None:
            print("✗ --fixture 를 주면 --fixture-episode 도 줘야 한다")
            return 2
        target = [p for p in eps if p.name == args.fixture_episode]
        if not target:
            target = [root / args.fixture_episode]
        print(f"\n== 계측기 검정 — {args.fixture_episode} ==")
        res = run_episode(target[0], args.draft)
        ok, msg = check_fixture(res, read_reference(args.fixture))
        print(("✓ 통과  " if ok else "✗ 실패  ") + msg)
        if not ok:
            print("\n기준 파일을 재현하지 못했다. **아무것도 쓰지 않는다.**")
            print("원본 tools/gripper_gap.py 를 받아서 그것으로 돌린다.")
            return 1
        if args.dry_run:
            print("\n--dry-run 이라 여기서 멈춘다.")
            return 0

    n_frames = n_failed = 0
    per_ep: list[tuple[str, int, int, float, float, int]] = []
    for i, ep in enumerate(eps, 1):
        if args.skip_existing and (ep / args.out_name).exists():
            continue
        res = run_episode(ep, args.draft)
        gaps = [r.gap_mm for r in res if r.gap_mm is not None]
        failed = sum(1 for r in res if r.status == STATUS_FAILED)
        n_frames += len(res)
        n_failed += failed
        if not args.dry_run:
            write_csv(ep, res, args.out_name)
        if gaps:
            g = np.array(gaps)
            th = (g.max() + g.min()) / 2.0
            closed = int((g < th).sum())
            per_ep.append((ep.name, len(res), failed, float(g.min()), float(g.max()), closed))
        else:
            per_ep.append((ep.name, len(res), failed, float("nan"), float("nan"), 0))
        if i % 10 == 0 or i == len(eps):
            print(f"  {i}/{len(eps)}", flush=True)

    print(f"\n총 {n_frames} 프레임 · 검출 실패 {n_failed} "
          f"({n_failed / max(n_frames, 1) * 100:.2f}%)")
    gmin = np.nanmin([p[3] for p in per_ep])
    gmax = np.nanmax([p[4] for p in per_ep])
    closed_frac = sum(p[5] for p in per_ep) / max(n_frames - n_failed, 1)
    print(f"gap 전체 범위 {gmin:.1f} ~ {gmax:.1f} mm")
    print(f"닫힘 프레임 비율 {closed_frac:.3f} "
          f"(BCE pos_weight 는 {(1 - closed_frac) / max(closed_frac, 1e-9):.3f} 가 된다)")

    worst = sorted(per_ep, key=lambda p: -p[2])[:5]
    print("\n검출 실패 많은 편:")
    for name, n, f, lo, hi, _ in worst:
        print(f"  {name}  {f}/{n}  gap {lo:.1f}~{hi:.1f}mm")

    flat = [p for p in per_ep if not np.isnan(p[3]) and (p[4] - p[3]) < 5.0]
    if flat:
        print(f"\n⚠️ gap 변화가 5mm 미만인 편 {len(flat)}개 — 폐쇄 이벤트가 없다:")
        for name, n, f, lo, hi, _ in flat[:10]:
            print(f"  {name}  {lo:.1f}~{hi:.1f}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
