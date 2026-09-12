"""Fixture — does the data-derived gripper convention reproduce the config one?
fixture — 데이터에서 뽑은 그리퍼 규약이 설정에서 파생된 값을 재현하는가?

왜 있는가 (2026-09-12).

`policy/bc.py` 의 `_GRIP_MID` 는 `grasp.open_cmd`/`close_cmd` 에서 파생된다. 그건 **시뮬**
값이다. 실물 UMI 시연은 열림 +0.2964 · 닫힘 -0.3414 이고 **둘 다 -0.4744 보다 위**라,
그 임계값을 그대로 쓰면 실물 프레임의 닫힘 라벨이 **0/76** 이 된다 🟢.

그래서 임계값을 데이터에서 뽑도록 바꿨다. 그 변경이 **기존 시뮬 결과를 건드리지 않는다**는
것을 여기서 검정한다 — 시뮬 데이터에 적용하면 설정 파생값이 그대로 나와야 한다.
안 나오면 기존 6/69/25 와 나란히 놓을 수 없고, 이 변경은 되돌려야 한다.

사용:
    python tools/check_gripper_norms.py --data out/dagger_merged_20260910_092947
    python tools/check_gripper_norms.py --data ... --gripper-csv <실물 gripper csv>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract.episode import read_episode  # noqa: E402
from policy.bc import (  # noqa: E402
    gripper_command_norms,
    gripper_index,
    gripper_norms_from_data,
    training_target,
)

# 시뮬 데이터에서 데이터 파생값이 설정 파생값과 이만큼 안에 들어와야 한다.
# 시뮬 명령은 정확히 두 값뿐이므로 분위수가 그 두 값을 그대로 집어야 정상이다.
SIM_TOLERANCE = 1e-6


def _actions(root: Path) -> torch.Tensor:
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"에피소드가 없다: {root}")
    arrs = [read_episode(f).action for f in files]
    return torch.from_numpy(np.concatenate(arrs, axis=0)).float()


def _gap_csv_to_norm(path: Path) -> torch.Tensor:
    """실물 gripper csv 의 gap_m 을 계약 정규화 값으로 바꾼다.
    gap -> 관절각은 `grasp.gap_curve`(행이 [rad, cm]) 역변환이고, 관절각 -> [-1,1] 은
    계약 공식이다. 여기 리터럴은 없다."""
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "so101.yaml"
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    lo, hi = (float(v) for v in cfg["joints"][gripper_index()]["range_rad"])
    tab = np.asarray(cfg["grasp"]["gap_curve"], dtype=float)
    ang, gaps_m = tab[:, 0], tab[:, 1] / 100.0
    vals = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("gap_m"):
                continue
            g = float(row["gap_m"])
            if g < gaps_m[0] or g > gaps_m[-1]:
                continue
            rad = float(np.interp(g, gaps_m, ang))
            vals.append(2.0 * (rad - lo) / (hi - lo) - 1.0)
    if not vals:
        raise ValueError(f"쓸 수 있는 gap 행이 없다: {path}")
    return torch.tensor(vals, dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="계약 에피소드 .npz 디렉터리 (시뮬)")
    ap.add_argument("--gripper-csv", type=Path, default=None,
                    help="실물 gripper csv. 주면 실물 쪽도 함께 본다")
    args = ap.parse_args()

    ref_open, ref_close = gripper_command_norms()
    ref_mid = (ref_open + ref_close) / 2.0
    g = gripper_index()
    fails: list[str] = []

    print("설정 파생 (grasp.open_cmd / close_cmd)")
    print(f"  열림 {ref_open:+.6f} · 닫힘 {ref_close:+.6f} · 임계값 {ref_mid:+.6f}\n")

    acts = _actions(args.data)
    d_open, d_close, d_mid = gripper_norms_from_data(acts[:, g])
    print(f"시뮬 데이터 파생 ({args.data}, {acts.shape[0]} 스텝)")
    print(f"  열림 {d_open:+.6f} · 닫힘 {d_close:+.6f} · 임계값 {d_mid:+.6f}")
    diffs = {"열림": abs(d_open - ref_open), "닫힘": abs(d_close - ref_close),
             "임계값": abs(d_mid - ref_mid)}
    for k, v in diffs.items():
        flag = "OK" if v <= SIM_TOLERANCE else "FAIL"
        if v > SIM_TOLERANCE:
            fails.append(f"시뮬 {k} 차이 {v:.3e} > {SIM_TOLERANCE:.0e}")
        print(f"  {k:5s} 차이 {v:.3e}  {flag}")

    # 라벨이 실제로 같은가 -- 값이 아니라 결과를 본다.
    sts = torch.zeros_like(acts)
    t_ref = training_target(acts, sts, "joint_delta_gripper_binary", None)[:, g]
    t_dat = training_target(acts, sts, "joint_delta_gripper_binary", d_mid)[:, g]
    n_diff = int((t_ref != t_dat).sum())
    print(f"  이진 라벨 불일치 {n_diff}/{t_ref.numel()}  " + ("OK" if n_diff == 0 else "FAIL"))
    if n_diff:
        fails.append(f"시뮬 이진 라벨이 {n_diff} 개 달라진다 — 기존 결과와 비교 불가")
    print(f"  닫힘 비율 {float(t_dat.mean()):.3f}\n")

    if args.gripper_csv is not None:
        real = _gap_csv_to_norm(args.gripper_csv)
        r_open, r_close, r_mid = gripper_norms_from_data(real)
        print(f"실물 데이터 파생 ({args.gripper_csv}, {real.numel()} 프레임)")
        print(f"  열림 {r_open:+.6f} · 닫힘 {r_close:+.6f} · 임계값 {r_mid:+.6f}")
        pad = torch.zeros((real.numel(), acts.shape[1]), dtype=torch.float32)
        pad[:, g] = real
        zero = torch.zeros_like(pad)
        n_ref = int(training_target(pad, zero, "joint_delta_gripper_binary", None)[:, g].sum())
        n_dat = int(training_target(pad, zero, "joint_delta_gripper_binary", r_mid)[:, g].sum())
        print(f"  닫힘 라벨 — 설정 임계값 {n_ref}/{real.numel()} · 데이터 임계값 {n_dat}/{real.numel()}")
        if n_ref != 0:
            print("  ⚠️ 설정 임계값으로도 0 이 아니다 — 이 문서의 전제를 다시 확인하라")
        if n_dat == 0 or n_dat == real.numel():
            fails.append(f"실물 닫힘 라벨이 {n_dat}/{real.numel()} 로 한쪽에 몰렸다")
        print(f"  닫힘 비율 {n_dat / real.numel():.3f} "
              f"→ pos_weight {(real.numel() - n_dat) / max(n_dat, 1):.3f}\n")

    if fails:
        print("✗ 실패 — 이 변경을 쓰지 않는다:")
        for f in fails:
            print("   " + f)
        return 1
    print("✓ 통과 — 시뮬 결과는 그대로이고 실물 라벨이 살아난다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
