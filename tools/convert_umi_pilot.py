"""ARCore bundles -> RawEpisode files. PILOT. 이슈 29 는 정의로 우회한다.

    python tools/convert_umi_pilot.py \
        --bundle-root /c/Users/SSAFY/Downloads/학습데이터 \
        --out out/umi_raw_pilot_0912 \
        --anchor-pos 0.22 0.0 0.10 \
        --limit 1

`--home-pinch` 에 기본값을 주지 않는다. 근사값이 조용히 정본이 되는 것을 막는다.
고를 때 근거: 도달 가능 영역 실측 🟢 — x 범위 [-0.03, 0.34], 최대 연속 직사각형 15x35cm.
시연 궤적 전체가 그 안에 들어가야 IK 가 풀린다. 이 도구가 들어가는지 검사해서 알려준다.

다음 단계는 기존 `tools/convert_umi.py` 가 그대로 받는다 (IK -> 계약 에피소드).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from umi.arcore_pilot import HANDEYE_MEASURED, PilotCalib, to_raw_pilot  # noqa: E402
from umi.raw import RawError, validate_raw, write_raw  # noqa: E402

# 도달 가능 영역 실측 🟢 (configs/so101.yaml 주석 · MEASURE_grasp_0827).
# 궤적이 여기 밖으로 나가면 IK 가 못 푼다 -- 변환 전에 알려준다.
REACH_X = (-0.03, 0.34)
REACH_R = 0.30          # 베이스에서 대략 반경 0.3m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--anchor-pos", type=float, nargs=3, required=True,
                    metavar=("X", "Y", "Z"),
                    help="로봇 베이스 기준 앵커 좌표 [m]. 기본값 없음 -- 근거를 갖고 고른다")
    ap.add_argument("--anchor-at", choices=("grasp", "first"), default="grasp",
                    help="궤적의 어느 지점을 앵커에 박을지. grasp=폐쇄 시점(권장)")
    ap.add_argument("--gripper-csv", type=str, default="gripper.csv",
                    help="번들 안의 그리퍼 파일명. 우리 재구현은 gripper_reimpl.csv")
    ap.add_argument("--image-size", type=int, default=224,
                    help="정사각형 축소 크기. 0 이면 원본 유지 (71편이면 약 33GB)")
    ap.add_argument("--calibration-id", type=str, default="pilot_handeye_0912",
                    help="빈 값이면 validate_raw 가 실기록을 거부한다")
    ap.add_argument("--trim-after-grasp", type=int, default=15,
                    help="폐쇄 시점 + 이만큼 프레임에서 자른다. -1 이면 자르지 않는다. "
                         "D-AI-22: 정책은 집기까지만 배운다")
    ap.add_argument("--pre-grasp-frames", type=int, default=45,
                    help="폐쇄 이전 이만큼만 남긴다 (30Hz 기준 45=1.5초). -1 이면 안 자른다")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N 편만 (0=전부)")
    ap.add_argument("--skill-id", type=str, default=None)
    args = ap.parse_args()

    eps = sorted(p for p in args.bundle_root.glob("rec_*") if (p / "frames").is_dir())
    if args.limit:
        eps = eps[:args.limit]
    if not eps:
        print(f"✗ 번들이 없다: {args.bundle_root}")
        return 2

    calib = PilotCalib(
        t_cam_pinch=HANDEYE_MEASURED,
        anchor_pos=np.asarray(args.anchor_pos, dtype=np.float64),
        calibration_id=args.calibration_id,
        anchor_at=args.anchor_at,
    )
    print(f"번들 {len(eps)}편 · 앵커 {args.anchor_at} @ {tuple(args.anchor_pos)} m")
    print("T_cam->pinch (MEASURE_handeye_from_video_0912) =")
    for r in HANDEYE_MEASURED:
        print("   [" + "  ".join(f"{v:+.4f}" for v in r) + "]")
    print()

    ok = fail = 0
    out_of_reach: list[tuple[str, float, float]] = []
    gap_missing: list[tuple[str, int, int]] = []

    for i, ep in enumerate(eps, 1):
        try:
            raw = to_raw_pilot(
                ep, calib,
                gripper_csv=args.gripper_csv,
                image_size=(args.image_size or None),
                skill_id=args.skill_id,
                trim_after_grasp=(None if args.trim_after_grasp < 0 else args.trim_after_grasp),
                pre_grasp_frames=(None if args.pre_grasp_frames < 0 else args.pre_grasp_frames),
            )
        except (RawError, KeyError, ValueError, FileNotFoundError) as exc:
            fail += 1
            print(f"  ✗ {ep.name}: {type(exc).__name__} {exc}")
            continue

        problems = validate_raw(raw)
        if problems:
            fail += 1
            print(f"  ✗ {ep.name}: raw 스키마 위반 {len(problems)}건")
            for p in problems[:3]:
                print(f"      {p}")
            continue

        p = raw.eef_pos
        r_xy = np.hypot(p[:, 0], p[:, 1])
        outside = int(((p[:, 0] < REACH_X[0]) | (p[:, 0] > REACH_X[1]) |
                       (r_xy > REACH_R)).sum())
        if outside:
            out_of_reach.append((ep.name, outside / len(p), float(r_xy.max())))

        n_nan = int(np.isnan(raw.gripper_gap_m).sum())
        if n_nan:
            gap_missing.append((ep.name, n_nan, len(raw.gripper_gap_m)))

        write_raw(raw, args.out)
        ok += 1
        if i % 10 == 0 or i == len(eps):
            print(f"  {i}/{len(eps)}  (성공 {ok} · 실패 {fail})", flush=True)

    print(f"\n변환 {ok}편 성공 · {fail}편 실패 → {args.out}")

    if out_of_reach:
        print(f"\n⚠️ 도달 가능 영역 밖 프레임이 있는 편 {len(out_of_reach)}개 "
              f"(x {REACH_X} · 반경 {REACH_R}m 기준)")
        for name, frac, rmax in sorted(out_of_reach, key=lambda t: -t[1])[:8]:
            print(f"   {name}  {frac:.0%}  최대반경 {rmax:.3f}m")
        print("   → 앵커를 옮기거나 창을 줄여야 IK 가 풀린다. "
              "**이 상태로 convert_umi 를 돌리면 수용률이 떨어진다**")
    else:
        print("\n✓ 모든 궤적이 도달 가능 영역 안이다")

    if gap_missing:
        tot_n = sum(m[1] for m in gap_missing)
        tot_d = sum(m[2] for m in gap_missing)
        print(f"\ngap 결측 {tot_n}/{tot_d} 프레임 ({tot_n / max(tot_d, 1):.1%}) — "
              f"T·X 상태다. 채우지 않았고 변환기가 그 스텝을 폐기한다")

    print("\n다음: tools/convert_umi.py --raw <위 out> --out datasets/umi_pilot_0912")
    print("⚠️ 이 데이터는 이슈 29 를 **정의로 우회**한 것이다. "
          "작업대 기준이 생기면 재변환한다 (meta.notes 에 박혀 있다)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
