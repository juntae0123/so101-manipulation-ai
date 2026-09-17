"""Convert real UMI demos (v10 relative chunks) into an official-UMI zarr dataset.
실 UMI 시연(v10 상대 청크)을 공식 UMI zarr 데이터셋으로 변환한다.

무엇을 하나
-----------
v10 은 절대 pose 를 담지 않는다. `action[i,0]` 을 연쇄해 상대 궤적을 복원하고,
**파지 순간을 시뮬 파지 pose 에 정렬**해 로봇 베이스 프레임 좌표를 만든다.

    T_base[t] = T_grasp_sim @ inv(chain[closure]) @ chain[t]

⚠️ 이것은 **실물 베이스 캘리브레이션이 아니다.** simulation-only grasp alignment 이고
   출력 provenance 에 그렇게 기록된다.

⚠️ 제품 기조는 **결과 모방**이다. 사람 궤적을 그대로 재현하지 않는다.
   그래서 정렬 기준이 시연 시작점이 아니라 파지 순간이다.

확정된 규약 (2026-09-17 스윕 🟢)
- `action_columns` = x,y,z + r0(3) + r1(3) + gap_m
- **r0,r1 은 회전행렬의 행이다.** 열로 읽으면 중앙오차 4mm 가 조용히 남는다
- 합성은 `T_next = T_cur @ A_relative`
- `source_row` 간격이 균일하지 않다. 시각은 `observation_timestamp` 에서 읽는다

산출물
------
    <out>.zarr.zip                공식 UMI 레이아웃
    <out>.provenance.json         정렬·규약·분할·검증 결과
    <out>.split.json              56/14 에피소드 분할 (시드 고정)

Usage
-----
  python convert_v10_to_umi.py --selftest
  python convert_v10_to_umi.py \
      --dataset ~/S15P21A103_umi/AI/datasets/umi_real_relative_20260911_v10 \
      --task ~/handoff/configs/can_side.yaml \
      --out ~/handoff/outputs/ds_real56 --split-seed 42 --holdout 14
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

GAP_TOL_M = 0.002
ACTION_COLUMNS = ["x_m", "y_m", "z_m", "r0x", "r0y", "r0z", "r1x", "r1y", "r1z", "gap_m"]


# ── 회전 ─────────────────────────────────────────────────────────────────

def rot6d_to_matrix(r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """Gram-Schmidt a 6D rotation representation into a 3x3 matrix.
    6D 회전 표현을 그람-슈미트로 3x3 행렬로. r0,r1 은 **행**이다."""
    b0 = np.asarray(r0, dtype=np.float64)
    n0 = np.linalg.norm(b0)
    if n0 < 1e-9:
        raise ValueError(f"r0 크기가 0에 가깝다: {n0:.3e}")
    b0 = b0 / n0
    a1 = np.asarray(r1, dtype=np.float64)
    b1 = a1 - np.dot(b0, a1) * b0
    n1 = np.linalg.norm(b1)
    if n1 < 1e-9:
        raise ValueError(f"r0 와 r1 이 평행하다 (잔차 {n1:.3e})")
    b1 = b1 / n1
    return np.column_stack([b0, b1, np.cross(b0, b1)]).T     # 행 기준


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector. 회전행렬을 축각 벡터로."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def row_to_transform(row: np.ndarray) -> np.ndarray:
    """One 10-column row -> 4x4 transform (gap ignored). 10열 행 → 4x4."""
    T = np.eye(4)
    T[:3, :3] = rot6d_to_matrix(row[3:6], row[6:9])
    T[:3, 3] = row[0:3]
    return T


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


# ── 궤적 ─────────────────────────────────────────────────────────────────

def reconstruct_chain(action: np.ndarray) -> np.ndarray:
    """Chain action[i,0] into transforms relative to the first row.
    action[i,0] 을 연쇄해 첫 행 기준 변환열을 만든다. (N+1,4,4)"""
    n = action.shape[0]
    out = np.zeros((n + 1, 4, 4))
    out[0] = np.eye(4)
    for i in range(n):
        out[i + 1] = out[i] @ row_to_transform(action[i, 0])
    return out


def chain_consistency(action: np.ndarray, chain: np.ndarray) -> dict:
    """Cross-check the chain against multi-step targets. 복원을 다단계 타깃과 대조."""
    pe, re_, n = [], [], 0
    N, H = action.shape[0], action.shape[1]
    for i in range(N):
        for k in range(1, H):
            j = i + k + 1
            if j >= chain.shape[0]:
                continue
            pred = chain[i] @ row_to_transform(action[i, k])
            pe.append(float(np.linalg.norm(pred[:3, 3] - chain[j][:3, 3])) * 1000)
            re_.append(geodesic_deg(pred[:3, :3], chain[j][:3, :3]))
            n += 1
    if n == 0:
        return {"comparisons": 0, "position_max_mm": None, "rotation_max_deg": None}
    return {"comparisons": n, "position_max_mm": float(np.max(pe)),
            "rotation_max_deg": float(np.max(re_))}


def closure_row(z, tol: float = GAP_TOL_M) -> tuple[int, float]:
    """First row where the jaws stop closing — that is where the object was.
    턱이 닫힘을 멈춘 첫 행. 물체가 있던 자리다. (행, 그때의 gap[m])"""
    gap = np.asarray(z["proprio"], dtype=np.float64)[:, -1, 9]
    if gap.size == 0:
        raise ValueError("proprio 가 비었다")
    c = int(np.argmax(gap <= float(gap.min()) + tol))
    return c, float(gap[c])


def sim_grasp_pose(env, seed: int) -> np.ndarray:
    """The sim's own side-grasp TCP pose for the object at this seed.
    이 시드에서 물체를 측면 파지할 때 시뮬이 쓰는 TCP pose.
    `simulation/expert.py: run_side_expert` 와 같은 식이다."""
    env.reset(seed)
    obj = env.data.body("object").xpos.copy()
    ap = np.r_[obj[:2] - np.array([0.04, 0.0]), 0.0]
    n = np.linalg.norm(ap)
    if n < 1e-9:
        raise ValueError("물체가 접근 기준점과 같은 위치다")
    ap = ap / n
    T = np.eye(4)
    T[:3, :3] = np.column_stack([np.cross([0, 0, 1], ap), [0, 0, 1], ap])
    T[:3, 3] = obj + ap * 0.013
    return T


# ── 자체 검증 ────────────────────────────────────────────────────────────

def selftest() -> int:
    """Verify the maths before trusting the conversion. 변환을 믿기 전에 검증한다."""
    rng = np.random.default_rng(0)
    bad = 0

    errs = []
    for _ in range(2000):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        errs.append(np.abs(rot6d_to_matrix(Q[0, :], Q[1, :]) - Q).max())
    print(f"[1] rot6d(행 규약) 왕복 2000회 최대오차 {max(errs):.3e}", end="  ")
    print("OK" if max(errs) < 1e-9 else "!! 실패"); bad += max(errs) >= 1e-9

    N, H = 12, 8
    step = np.eye(4); step[:3, 3] = [0.01, -0.005, 0.002]
    a = math.radians(2.0)
    step[:3, :3] = np.array([[math.cos(a), -math.sin(a), 0],
                             [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    truth = [np.eye(4)]
    for _ in range(N + H):
        truth.append(truth[-1] @ step)
    act = np.zeros((N, H, 10))
    for i in range(N):
        for k in range(H):
            rel = np.linalg.inv(truth[i]) @ truth[i + k + 1]
            R = rel[:3, :3]
            act[i, k] = np.r_[rel[:3, 3], R[0, :], R[1, :], 0.05]
    chain = reconstruct_chain(act)
    e = max(np.abs(chain[i] - truth[i]).max() for i in range(N + 1))
    print(f"[2] 연쇄 복원 최대오차 {e:.3e}", end="  ")
    print("OK" if e < 1e-9 else "!! 실패"); bad += e >= 1e-9

    cc = chain_consistency(act, chain)
    print(f"[3] 교차검증 {cc['comparisons']}건 위치최대 {cc['position_max_mm']:.6f}mm", end="  ")
    ok3 = cc["comparisons"] > 0 and cc["position_max_mm"] < 1e-6
    print("OK" if ok3 else "!! 실패"); bad += not ok3

    broken = act.copy(); broken[3, 4, 0] += 0.05
    cb = chain_consistency(broken, chain)
    caught = cb["position_max_mm"] > 40.0
    print(f"[4] 고의 손상 감지 {cb['position_max_mm']:.3f}mm", end="  ")
    print("OK" if caught else "!! 실패 — 망가진 데이터를 못 잡는다"); bad += not caught

    # 정렬이 파지 pose 를 정확히 맞추는지
    Tg = np.eye(4); Tg[:3, 3] = [0.4, 0.0, 0.05]
    c = 5
    Talign = Tg @ np.linalg.inv(chain[c])
    err = np.abs((Talign @ chain[c]) - Tg).max()
    print(f"[5] 파지 정렬 오차 {err:.3e}", end="  ")
    print("OK" if err < 1e-12 else "!! 실패"); bad += err >= 1e-12

    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


# ── 본체 ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset")
    ap.add_argument("--task")
    ap.add_argument("--out")
    ap.add_argument("--anchor-seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--holdout", type=int, default=14)
    ap.add_argument("--consistency-mm", type=float, default=1.0,
                    help="복원 교차검증 허용 오차[mm]. 넘으면 그 편을 버린다")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    for need in ("dataset", "task", "out"):
        if not getattr(a, need):
            raise SystemExit(f"!! --{need} 가 필요하다 (또는 --selftest)")

    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 변환하지 않는다")
    print()

    from umi_adapter.upstream import enable
    enable()
    import zarr
    from simulation.env import PickEnv

    d = Path(a.dataset).expanduser()
    meta = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
    print(f"입력 {meta['schema']} · {meta['n_episodes']}편 · rate {meta['rate_hz']}Hz "
          f"· horizon {meta['action_horizon']}")

    env = PickEnv(task_path=str(Path(a.task).expanduser()))
    T_grasp = sim_grasp_pose(env, a.anchor_seed)
    env.close()
    print(f"시뮬 파지 pose (seed {a.anchor_seed}) {np.round(T_grasp[:3,3],4)}\n")

    rgb, pos, rot, grip, ends = [], [], [], [], []
    kept, dropped, gaps = [], [], []
    total = 0

    for name in meta["episodes"]:
        f = d / f"{name}.npz"
        if not f.exists():
            dropped.append({"episode": name, "reason": "npz 없음"}); continue
        z = np.load(f)
        act = np.asarray(z["action"], dtype=np.float64)
        img = np.asarray(z["image"])                      # (N,2,3,224,224) uint8
        pro = np.asarray(z["proprio"], dtype=np.float64)  # (N,2,10)
        if act.shape[2] != len(ACTION_COLUMNS):
            dropped.append({"episode": name, "reason": f"action 열 {act.shape[2]}"}); continue
        if img.shape[0] != act.shape[0] or pro.shape[0] != act.shape[0]:
            dropped.append({"episode": name, "reason": "행 수 불일치"}); continue

        chain = reconstruct_chain(act)
        cc = chain_consistency(act, chain)
        if cc["comparisons"] == 0 or cc["position_max_mm"] > a.consistency_mm:
            dropped.append({"episode": name, "reason": "복원 교차검증 실패",
                            "consistency": cc}); continue

        c, gmm = closure_row(z)
        gaps.append(gmm * 1000)
        T_align = T_grasp @ np.linalg.inv(chain[c])

        n = act.shape[0]
        for i in range(n):
            T = T_align @ chain[i]
            frame = img[i, -1]                            # 현재 관측 (3,224,224)
            rgb.append(np.ascontiguousarray(frame.transpose(1, 2, 0)))   # HWC
            pos.append(T[:3, 3].astype(np.float32))
            rot.append(matrix_to_rotvec(T[:3, :3]).astype(np.float32))
            grip.append(np.array([pro[i, -1, 9]], dtype=np.float32))
        total += n
        ends.append(total)
        kept.append(name)

    if not kept:
        raise SystemExit("!! 변환된 에피소드가 0편이다")

    gs = sorted(gaps)
    spread = gs[-1] - gs[0]
    print(f"닫힘 gap(=물체 폭) 중앙 {gs[len(gs)//2]:.1f}mm 범위 {gs[0]:.1f}~{gs[-1]:.1f}mm")
    if spread > 15.0:
        raise SystemExit(f"!! 물체 폭이 {spread:.1f}mm 흩어진다. 닫힘 검출이 틀렸을 수 있다. "
                         f"변환하지 않는다")

    # 고정 분할 — 학습 시드가 바뀌어도 이 분할은 안 바뀐다
    rng = np.random.default_rng(a.split_seed)
    order = list(kept)
    rng.shuffle(order)
    hold = sorted(order[:a.holdout])
    train = sorted(order[a.holdout:])

    out = Path(a.out).expanduser()
    store = zarr.ZipStore(str(out) + ".zarr.zip", mode="w")
    root = zarr.group(store=store)
    data = root.create_group("data")
    rgb_a = np.stack(rgb)
    data.create_dataset("camera0_rgb", data=rgb_a, chunks=(1,) + rgb_a.shape[1:], dtype="uint8")
    data.create_dataset("robot0_eef_pos", data=np.stack(pos), dtype="float32")
    data.create_dataset("robot0_eef_rot_axis_angle", data=np.stack(rot), dtype="float32")
    data.create_dataset("robot0_gripper_width", data=np.stack(grip), dtype="float32")
    sp = np.concatenate([np.stack(pos), np.stack(rot)], axis=1).astype(np.float64)
    data.create_dataset("robot0_demo_start_pose", data=sp, dtype="float64")
    data.create_dataset("robot0_demo_end_pose", data=sp, dtype="float64")
    root.create_group("meta").create_dataset("episode_ends", data=np.array(ends, dtype=np.int64))
    store.close()

    prov = {
        "source": str(d), "source_schema": meta["schema"],
        "alignment": "simulation-only grasp alignment "
                     "(T_base = T_grasp_sim @ inv(chain[closure]) @ chain[t]) "
                     "— NOT a physical robot-base calibration",
        "anchor_seed": a.anchor_seed,
        "rotation_convention": "r0,r1 = first two ROWS of R (2026-09-17 스윕 확정)",
        "composition": "T_next = T_cur @ A_relative",
        "rate_hz": meta["rate_hz"], "source_action_horizon": meta["action_horizon"],
        "episodes_kept": len(kept), "frames": total,
        "dropped": dropped,
        "closure_gap_mm": {"median": gs[len(gs) // 2], "min": gs[0], "max": gs[-1]},
        "split": {"seed": a.split_seed, "train": len(train), "holdout": len(hold)},
        "limitations": [
            "물체 배치 시드 1개만 사용",
            "접근 경로 미측정 ([ROS] MoveIt 범위)",
            "실물 베이스 캘리브레이션 아님",
        ],
    }
    Path(str(out) + ".provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(str(out) + ".split.json").write_text(
        json.dumps({"seed": a.split_seed, "train": train, "holdout": hold},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n변환 {len(kept)}편 · 프레임 {total} · 버림 {len(dropped)}편")
    if dropped:
        print("  버린 이유:", {x["reason"] for x in dropped})
    print(f"분할 고정 (시드 {a.split_seed}) — 학습 {len(train)}편 / 홀드아웃 {len(hold)}편")
    print(f"→ {out}.zarr.zip")
    print(f"→ {out}.provenance.json · {out}.split.json")
    print("⚠️ simulation-only grasp alignment. 실물 캘리브레이션이 아니다.")


if __name__ == "__main__":
    main()
