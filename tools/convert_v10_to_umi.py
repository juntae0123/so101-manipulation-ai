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
from collections import Counter
from pathlib import Path

import numpy as np

GAP_TOL_M = 0.002
# 닫힘으로 인정하는 gap 총 이동량 하한. 실 시연 닫힘 gap 중앙 38.5mm [32.4, 43.5] (n=69)
# 대비 한참 아래로 잡아 정상 편을 떨구지 않으면서 "평평한 편"만 잡는다.
CLOSURE_MIN_DROP_M = 0.010
# 복원 교차검증 회전 허용치. 초판은 위치만 게이트에 썼다 (2026-09-20 보강).
CONSISTENCY_DEG_DEFAULT = 2.0
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


class ClosureNotFound(ValueError):
    """The demo never closed — refuse rather than calling row 0 the grasp.
    닫힌 적이 없는 시연. 0행을 파지 순간이라고 부르는 대신 거부한다."""


def closure_row(z, tol: float = GAP_TOL_M,
                min_drop_m: float = CLOSURE_MIN_DROP_M) -> tuple[int, float]:
    """First row where the jaws stop closing — that is where the object was.
    턱이 닫힘을 멈춘 첫 행. 물체가 있던 자리다. (행, 그때의 gap[m])

    ⚠️ 2026-09-20 보강 — 초판은 `argmax(gap <= min + tol)` 하나였다.
       gap 이 내내 평평하면 min ≈ gap[0] 이라 **말없이 0 을 돌려준다.** 그러면
       `T_align = T_grasp @ inv(chain[0])` 이 되어 **시연 첫 프레임이 시뮬 파지
       pose 에 붙고 전 궤적이 통째로 오프셋된 채** 학습 데이터가 된다.
       "없음"과 "괜찮음"이 같은 출력이 되는 정확한 형태다. 두 게이트를 건다.
    """
    gap = np.asarray(z["proprio"], dtype=np.float64)[:, -1, 9]
    if gap.size == 0:
        raise ClosureNotFound("proprio 가 비었다")
    drop = float(gap.max() - gap.min())
    if drop < min_drop_m:
        raise ClosureNotFound(
            f"gap 총 이동량이 {drop*1000:.2f}mm 뿐이다 (하한 {min_drop_m*1000:.1f}mm). "
            "닫힌 적이 없거나 신호가 죽었다 — 파지 순간을 특정할 수 없다")
    c = int(np.argmax(gap <= float(gap.min()) + tol))
    if c == 0 or c == gap.size - 1:
        raise ClosureNotFound(
            f"닫힘 최소점이 경계 행({c}/{gap.size - 1})이다 — 구간이 잘렸을 수 있다")
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

    # [6] 전치 판별행 — 행/열 규약이 뒤집히면 잡아야 한다. 초판 selftest 는 자기가
    #     행으로 만들어 넣고 행으로 읽어 비교하는 동어반복이라, 규약이 틀려도 통과했다.
    act_T = act.copy()
    for i in range(N):
        for k in range(H):
            R = rot6d_to_matrix(act[i, k, 3:6], act[i, k, 6:9]).T      # 전치해 넣는다
            act_T[i, k, 3:6], act_T[i, k, 6:9] = R[0, :], R[1, :]
    # ⚠️ 자기 자신과 비교하면 안 된다 — 전치로 넣고 전치로 읽으면 늘 일치한다(동어반복).
    #    **정답 truth** 와 비교해야 규약 위반이 드러난다.
    chain_T = reconstruct_chain(act_T)
    dev = max(geodesic_deg(chain_T[i][:3, :3], truth[i][:3, :3]) for i in range(N + 1))
    caught6 = dev > 1.0
    print(f"[6] 전치 입력 감지 — 정답 대비 회전편차 {dev:.3f}deg", end="  ")
    print("OK" if caught6 else "!! 실패 — 행/열이 뒤집혀도 정답과 같게 나온다")
    bad += not caught6

    # [7-9] closure_row — "닫힘 없음"과 "0행이 파지"를 가르는 게이트 (2026-09-20)
    def _mk(gap_seq):
        """gap 열만 채운 최소 proprio 스텁. z["proprio"][:, -1, 9] 만 읽힌다."""
        arr = np.zeros((len(gap_seq), 1, 10), dtype=np.float64)
        arr[:, 0, 9] = gap_seq

        class _Z:
            files = ["proprio"]

            def __getitem__(self, k):
                if k != "proprio":
                    raise KeyError(k)
                return arr

        return _Z()

    flat = _mk([0.088] * 20)
    try:
        closure_row(flat); ok7 = False
    except ClosureNotFound:
        ok7 = True
    print(f"[7] 평평한 gap 거부 (초판은 0행을 파지라 답했다)", end="  ")
    print("OK" if ok7 else "!! 실패"); bad += not ok7

    edge = _mk([0.030] + [0.088] * 19)          # 최소점이 0행
    try:
        closure_row(edge); ok8 = False
    except ClosureNotFound:
        ok8 = True
    print(f"[8] 경계 행 닫힘 거부", end="  ")
    print("OK" if ok8 else "!! 실패"); bad += not ok8

    good = _mk(list(np.linspace(0.075, 0.038, 12)) + [0.038] * 8)   # 정상 닫힘
    try:
        c9, g9 = closure_row(good)
        ok9 = 0 < c9 < 19 and abs(g9 - 0.038) < 1e-6
    except ClosureNotFound:
        c9, g9, ok9 = -1, -1, False
    print(f"[9] 정상 닫힘 통과 판별행 행 {c9} gap {g9*1000:.1f}mm", end="  ")
    print("OK" if ok9 else "!! 실패 — 전부 거부하는 게이트는 게이트가 아니다"); bad += not ok9

    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'} (검사 9건)")
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
    ap.add_argument("--only", choices=["all", "train", "holdout"], default="all",
                    help="분할 중 어느 쪽만 zarr 로 낼지. "
                         "**학습에는 train, 평가에는 holdout 을 쓴다.** "
                         "all 로 학습하면 홀드아웃이 학습에 섞인다")
    ap.add_argument("--consistency-mm", type=float, default=1.0,
                    help="복원 교차검증 허용 오차[mm]. 넘으면 그 편을 버린다")
    ap.add_argument("--consistency-deg", type=float, default=CONSISTENCY_DEG_DEFAULT,
                    help="복원 교차검증 회전 허용치[도]. 초판은 회전을 게이트에 안 썼다")
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

    # 분할을 **먼저** 정한다. 어느 편을 내보내든 분할은 같아야 하기 때문이다.
    # (변환 후에 나누면 --only 마다 분할이 달라질 수 있다)
    # ⚠️ 정렬 후 셔플한다. 초판은 JSON 의 배열 순서를 그대로 셔플해서, 데이터셋을
    #    재생성해 순서가 바뀌면 같은 --split-seed 로 다른 분할이 나왔다 (2026-09-20).
    all_eps = sorted(meta["episodes"])
    rng0 = np.random.default_rng(a.split_seed)
    shuffled = list(all_eps)
    rng0.shuffle(shuffled)
    hold_set = set(shuffled[:a.holdout])
    if a.only == "train":
        want = [e for e in all_eps if e not in hold_set]
    elif a.only == "holdout":
        want = [e for e in all_eps if e in hold_set]
    else:
        want = all_eps
    print(f"분할 시드 {a.split_seed} · 홀드아웃 {a.holdout}편 · 이번 출력 = {a.only} ({len(want)}편)")

    rgb, pos, rot, grip, ends = [], [], [], [], []
    start_pose, end_pose = [], []          # F3: 편별 시작/끝 pose (프레임 pose 가 아니다)
    kept, dropped, gaps, kept_cc, ep_start6, ts_stats = [], [], [], [], [], []
    total = 0

    for name in want:
        f = d / f"{name}.npz"
        if not f.exists():
            dropped.append({"episode": name, "reason": "npz 없음"}); continue
        z = np.load(f)
        act = np.asarray(z["action"], dtype=np.float64)
        img = np.asarray(z["image"])                      # (N,2,3,224,224) uint8
        pro = np.asarray(z["proprio"], dtype=np.float64)  # (N,2,10)
        if act.shape[2] != len(ACTION_COLUMNS):
            dropped.append({"episode": name, "reason": f"action 열 {act.shape[2]}"}); continue
        # proprio 도 검사한다. 아래에서 pro[:, -1, 9] 를 직접 인덱싱하므로, 레이아웃이
        # 바뀌면 gap 이 아닌 채널을 읽고도 IndexError 없이 진행한다 (2026-09-20 보강).
        if pro.ndim != 3 or pro.shape[2] < 10:
            dropped.append({"episode": name,
                            "reason": f"proprio 형상 {tuple(pro.shape)} — gap 열(9) 없음"}); continue
        if img.shape[0] != act.shape[0] or pro.shape[0] != act.shape[0]:
            dropped.append({"episode": name, "reason": "행 수 불일치"}); continue

        chain = reconstruct_chain(act)
        cc = chain_consistency(act, chain)
        if cc["comparisons"] == 0:
            dropped.append({"episode": name, "reason": "복원 교차검증 비교 0건",
                            "consistency": cc}); continue
        if cc["position_max_mm"] > a.consistency_mm:
            dropped.append({"episode": name, "reason": "복원 교차검증 위치 초과",
                            "consistency": cc}); continue
        # ⚠️ 초판은 rotation_max_deg 를 계산해 놓고 게이트에 쓰지 않았다 (2026-09-20).
        #    위치가 완벽하고 회전만 20도 틀어진 편이 1mm 게이트를 통과해 학습에 들어갔다.
        #    회전은 r0/r1 행-열 규약이 지배하는 축이고, 그 규약의 런타임 계측기가 이것뿐이다.
        if cc["rotation_max_deg"] > a.consistency_deg:
            dropped.append({"episode": name, "reason": "복원 교차검증 회전 초과",
                            "consistency": cc}); continue

        try:
            c, gmm = closure_row(z)
        except ClosureNotFound as exc:
            dropped.append({"episode": name, "reason": "닫힘 검출 실패",
                            "detail": str(exc)}); continue
        gaps.append(gmm * 1000)
        # ⚠️ docstring 은 "시각은 observation_timestamp 에서 읽는다"고 단언하는데
        #    초판 코드에는 그 문자열이 한 번도 없었다 (2026-09-20). 균일 레이트로 나가면서
        #    불균일을 기록조차 안 했다. 지금도 리샘플은 하지 않는다 — 다만 **잰다**.
        ts_stat = {"source": None, "note": "타임스탬프 필드 없음 — 균일 가정 미검증"}
        for key in ("observation_timestamp", "source_row"):
            if key in getattr(z, "files", []):
                v = np.asarray(z[key], dtype=np.float64).ravel()
                if v.size >= 2:
                    dv = np.diff(v)
                    ts_stat = {"source": key, "n": int(v.size),
                               "step_min": float(dv.min()), "step_max": float(dv.max()),
                               "step_median": float(np.median(dv)),
                               "uniform": bool(np.allclose(dv, dv[0]))}
                break
        ts_stats.append({"episode": name, **ts_stat})
        # 통과한 편의 잔차도 남긴다. 초판은 dropped 에만 실어서, 사후에
        # "얼마나 아슬아슬하게 통과했나"를 볼 수 없었다.
        kept_cc.append({"episode": name, "closure_row": c, "closure_gap_mm": gmm * 1000,
                        "position_max_mm": cc["position_max_mm"],
                        "rotation_max_deg": cc["rotation_max_deg"],
                        "comparisons": cc["comparisons"]})
        T_align = T_grasp @ np.linalg.inv(chain[c])

        n = act.shape[0]
        ep_pose = []
        for i in range(n):
            T = T_align @ chain[i]
            frame = img[i, -1]                            # 현재 관측 (3,224,224)
            rgb.append(np.ascontiguousarray(frame.transpose(1, 2, 0)))   # HWC
            p6 = np.r_[T[:3, 3], matrix_to_rotvec(T[:3, :3])]
            ep_pose.append(p6)
            pos.append(T[:3, 3].astype(np.float32))
            rot.append(matrix_to_rotvec(T[:3, :3]).astype(np.float32))
            grip.append(np.array([pro[i, -1, 9]], dtype=np.float32))
        # ⚠️ 초판은 프레임별 현재 pose 를 start 에도 end 에도 그대로 넣었다 (start==end==current).
        #    demo_start 기준 상대 pose 를 쓰는 경로에서는 전 프레임이 항등원이 된다 —
        #    학습은 정상으로 돌고 loss 도 내려가지만 조건 입력이 상수다 (2026-09-20).
        sp_ep, ep_ep = ep_pose[0], ep_pose[-1]
        start_pose.extend([sp_ep] * n)
        end_pose.extend([ep_ep] * n)
        ep_start6.append({"episode": name,
                          "start_pose_6dof": [round(float(v), 6) for v in sp_ep],
                          "end_pose_6dof": [round(float(v), 6) for v in ep_ep],
                          "frames": n})
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

    hold = sorted(e for e in all_eps if e in hold_set)
    train = sorted(e for e in all_eps if e not in hold_set)

    out = Path(a.out).expanduser()
    store = zarr.ZipStore(str(out) + ".zarr.zip", mode="w")
    root = zarr.group(store=store)
    data = root.create_group("data")
    rgb_a = np.stack(rgb)
    data.create_dataset("camera0_rgb", data=rgb_a, chunks=(1,) + rgb_a.shape[1:], dtype="uint8")
    data.create_dataset("robot0_eef_pos", data=np.stack(pos), dtype="float32")
    data.create_dataset("robot0_eef_rot_axis_angle", data=np.stack(rot), dtype="float32")
    data.create_dataset("robot0_gripper_width", data=np.stack(grip), dtype="float32")
    data.create_dataset("robot0_demo_start_pose",
                        data=np.stack(start_pose).astype(np.float64), dtype="float64")
    data.create_dataset("robot0_demo_end_pose",
                        data=np.stack(end_pose).astype(np.float64), dtype="float64")
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
        "episodes_requested": len(want), "episodes_kept": len(kept),
        "episodes_dropped": len(dropped), "frames": total,
        # ⚠️ 초판은 개수만 남겨서 "어느 편이 학습에 들어갔나"를 사후에 답할 수 없었다.
        #    같은 원본에서 트랙A 74편 / HW v4 76편 / 공통 64편이 나온 사고와 같은 모양이다.
        "episodes_kept_names": list(kept),
        "episode_ends": [int(x) for x in ends],
        "episode_ends_note": "episode_ends[i] 는 episodes_kept_names[i] 의 끝 프레임이다",
        "dropped": dropped,
        "dropped_by_reason": {k: v for k, v in sorted(Counter(
            x["reason"] for x in dropped).items(), key=lambda kv: -kv[1])},
        "kept_consistency": kept_cc,
        "timestamps": ts_stats,
        "closure_gap_mm": {"median": gs[len(gs) // 2], "min": gs[0], "max": gs[-1]},
        "only": a.only,
        "split": {"seed": a.split_seed, "train": len(train), "holdout": len(hold),
                  "note": f"분할은 원본 {len(all_eps)}편 전체에 대해 정해진다. --only 는 "
                          "그중 어느 쪽을 내보낼지만 고른다. 학습=train, 평가=holdout. "
                          "탈락 편은 여기서 빠지지 않으므로 분모로 쓰지 마라 — "
                          "실제로 zarr 에 들어간 것은 converted_this_run 이다"},
        "episode_start_pose_6dof": ep_start6,
        "limitations": [
            "물체 배치 시드 1개만 사용",
            "접근 경로 미측정 ([ROS] MoveIt 범위)",
            "실물 베이스 캘리브레이션 아님",
        ],
    }
    Path(str(out) + ".provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    # ⚠️ 초판 split.json 은 드랍을 반영하지 않았다. `--only holdout` 로 14편을 뽑을 때
    #    3편이 탈락해 zarr 가 11편이어도 holdout 14편이라고 말했고, 평가가 그걸 분모로
    #    삼으면 성공률 분모가 3편 부풀려진다 (2026-09-20).
    Path(str(out) + ".split.json").write_text(
        json.dumps({"seed": a.split_seed, "train": train, "holdout": hold,
                    "only": a.only,
                    "requested_this_run": list(want),
                    "converted_this_run": list(kept),
                    "dropped_this_run": [{"episode": x["episode"], "reason": x["reason"]}
                                         for x in dropped],
                    "note": "train/holdout 은 탈락 전 계획이다. 분모로 쓸 것은 "
                            "converted_this_run 이다"},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # ⚠️ 부분합이 전체와 같은지 확인한다. 폐기 경로를 하나만 세면 조용히 틀린다.
    if len(kept) + len(dropped) != len(want):
        raise SystemExit(f"!! 집계가 안 맞는다. 요청 {len(want)} != 통과 {len(kept)} + "
                         f"탈락 {len(dropped)}. 세지 않은 폐기 경로가 있다")
    print(f"\n변환 {len(kept)}/{len(want)}편 · 프레임 {total} · 버림 {len(dropped)}/{len(want)}편")
    for reason, cnt in sorted(Counter(x["reason"] for x in dropped).items(),
                              key=lambda kv: -kv[1]):
        print(f"  버림 {cnt}/{len(want)}편 — {reason}")
    nonuni = [t for t in ts_stats if t.get("uniform") is False]
    nots = [t for t in ts_stats if t.get("source") is None]
    print(f"시간축: 잰 편 {len(ts_stats) - len(nots)}/{len(ts_stats)} · "
          f"불균일 {len(nonuni)}편 (리샘플은 하지 않는다 — 기록만)")
    print(f"분할 고정 (시드 {a.split_seed}) — 학습 {len(train)}편 / 홀드아웃 {len(hold)}편")
    if a.only == "all":
        print("!! --only all 이다. 이 zarr 로 학습하면 홀드아웃이 학습에 섞인다. "
              "학습용은 --only train 으로 다시 뽑아라")
    print(f"→ {out}.zarr.zip")
    print(f"→ {out}.provenance.json · {out}.split.json")
    print("⚠️ simulation-only grasp alignment. 실물 캘리브레이션이 아니다.")


if __name__ == "__main__":
    main()
