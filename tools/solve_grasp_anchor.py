"""Where should the object sit so the grasp is reachable? Solved from the demos.
대상물을 어디에 놓아야 파지가 도달권에 드는가. 시연에서 직접 푼다.

Why this exists / 왜 만들었나
-----------------------------
`robot_base_alignment_applied: false` 다. 실물 베이스 정렬 좌표는 트랙 A(황도경)
소관이고 아직 못 받았다. 그동안 시연 궤적을 시뮬 홈 pose 에 박아 놓고 재면
목표점 대부분이 작업영역 밖에 떨어진다 (한 편에서 작업영역 안 3/80 🟢).
그 숫자를 "5자유도가 모자라다" 로 읽으면 틀린다. **놓는 자리를 안 정했을 뿐이다.**

이 도구는 정렬 좌표를 **기다리는 대신** 반대로 푼다 —
*"파지 순간이 도달권에 들려면 물체를 어디에 놓아야 하나"* 를 탐색한다.

⚠️ 이것은 실물 베이스 캘리브레이션이 **아니다.** 우리가 내는 **제안값**이고,
   실물에 올리기 전에 황도경·김현석 확인이 필요하다. 리포트 모든 행에 그렇게 적힌다.

무엇을 성공으로 보나 / What counts as success
---------------------------------------------
**파지 순간 전후만 본다.** 시연 전체 궤적이 아니다. 제품 기조가 결과 모방이므로
접근 경로는 로봇이 스스로 계획한다. 전체 궤적으로 재면 숫자가 통째로 부풀려진다
(2026-09-17 실증: 전체궤적 29.2% vs 파지기준 533/535 = 99.6%).

어떻게 / How
------------
한 후보 배치 (p, yaw) 에 대해 정렬 변환을 이렇게 만든다.

    T_align(p, yaw) = Trans(p) @ Rz(yaw) @ Trans(-g)      g = chain[closure] 의 위치

이러면 `T_align @ chain[closure]` 의 위치가 정확히 p 가 되고, 자세는 시연 자세를
월드 z 축으로 yaw 만큼 돌린 것이 된다. 그 다음은 **검증된 계측기를 그대로 쓴다** —
`check_real_traj_ik.check_episode(env, chain, T_align, ..., span=파지구간)`.

2단 탐색. 1단은 닫힘 pose 하나만 IK 로 찍어 후보를 거르고(싸다), 2단은 살아남은
후보에만 파지 구간 전체를 돌린다. 1단만 보고 결론 내지 않는다.

Usage
-----
  # [서버]
  ~/envs/handoff312/bin/python AI/tools/solve_grasp_anchor.py --selftest
  ~/envs/handoff312/bin/python AI/tools/solve_grasp_anchor.py --dataset ~/S15P21A103_umi/AI/datasets/umi_real_relative_20260911_v10 --task ~/handoff/configs/can_side.yaml --out ~/handoff/outputs/grasp_anchor.json

되돌리기 / Reverting
--------------------
이 파일을 지우면 끝이다. 데이터셋·설정·handoff 를 하나도 건드리지 않는다.
출력은 `--out` JSON 하나뿐이다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from check_real_traj_ik import (  # noqa: E402  — 검증된 계측기를 재사용한다
    ACTION_COLUMNS,
    check_episode,
    grasp_window,
    measure_reference_limits,
    reconstruct_relative_chain,
    step_durations,
)

# Measured reach envelope, side grasp, handoff SO-101, base_height 0. 🟢
# 도달 포락선 실측(측면 파지). 발명값이 아니다 — MEASURE_task2_envelope_0916.
ENVELOPE = {"x": (0.34, 0.46), "y": (-0.175, 0.175), "z": (0.027, 0.051)}


def rz(yaw_rad: float) -> np.ndarray:
    """Rotation about world z as a 4x4.
    월드 z 축 회전을 4x4 로."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return T


def align_transform(grasp_pos: np.ndarray, p: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Place this episode's grasp point at p, yawed by yaw_rad.
    이 편의 파지점을 p 에 놓고 yaw 만큼 돌리는 정렬 변환.

    T = Trans(p) @ Rz(yaw) @ Trans(-g).  순서를 바꾸면 위치가 틀어진다."""
    tp, tg = np.eye(4), np.eye(4)
    tp[:3, 3] = np.asarray(p, dtype=np.float64)
    tg[:3, 3] = -np.asarray(grasp_pos, dtype=np.float64)
    return tp @ rz(yaw_rad) @ tg


def load_episodes(dataset: Path, horizon: int, pre: int, post: int,
                  limit: int = 0) -> tuple[list[dict], list[dict]]:
    """Reconstruct every episode's chain and grasp window. Report both counts.
    편별 궤적과 파지 구간을 복원한다. 쓴 편수와 버린 편수를 같이 돌려준다."""
    meta = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    names = meta["episodes"][:limit] if limit else meta["episodes"]
    used, dropped = [], []
    for name in names:
        f = dataset / f"{name}.npz"
        if not f.exists():
            dropped.append({"episode": name, "reason": "npz 없음"})
            continue
        z = np.load(f)
        action = np.asarray(z["action"], dtype=np.float64)
        if action.shape[1] != horizon or action.shape[2] != len(ACTION_COLUMNS):
            dropped.append({"episode": name, "reason": f"action shape {action.shape}"})
            continue
        chain = reconstruct_relative_chain(action)
        lo, hi, closure = grasp_window(z, chain.shape[0], pre, post)
        if hi - lo < 2:
            dropped.append({"episode": name, "reason": f"파지 구간 {hi-lo+1}행"})
            continue
        used.append({"episode": name, "chain": chain, "span": (lo, hi),
                     "closure": closure, "dts": step_durations(z, chain.shape[0])})
    return used, dropped


def grid(n_x: int, n_y: int, n_z: int, n_yaw: int) -> list[tuple[np.ndarray, float]]:
    """Candidate placements inside the measured envelope.
    실측 포락선 안의 후보 배치."""
    xs = np.linspace(*ENVELOPE["x"], n_x)
    ys = np.linspace(*ENVELOPE["y"], n_y)
    zs = np.linspace(*ENVELOPE["z"], n_z)
    yaws = np.linspace(-np.pi, np.pi, n_yaw, endpoint=False)
    return [(np.array([x, y, z]), float(w))
            for x in xs for y in ys for z in zs for w in yaws]


def closure_reachable(env, ep: dict, T_align: np.ndarray, ik_tol: float) -> bool:
    """Stage 1: does the closure pose alone solve? Cheap filter, not a verdict.
    1단: 닫힘 pose 하나가 풀리는가. 싼 거름망이지 판정이 아니다."""
    import mujoco
    T = T_align @ ep["chain"][ep["closure"]]
    pos, R = T[:3, 3], T[:3, :3]
    try:
        q = env.ik(pos, R, seed=env.home_q, strict=False)
    except Exception:                                   # noqa: BLE001
        return False
    if q is None:
        return False
    lo, hi = env.limits[:, 0], env.limits[:, 1]
    if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
        return False
    env.ik_data.qpos[env.qids] = q
    mujoco.mj_forward(env.model, env.ik_data)
    return float(np.linalg.norm(env.tcp(env.ik_data)[:3, 3] - pos)) <= ik_tol


def selftest() -> int:
    """Geometry rows with known answers, plus deliberately wrong inputs.
    정답을 아는 기하 검정 + 고의로 틀린 입력."""
    ok = tot = 0

    def check(label: str, cond: bool) -> None:
        nonlocal ok, tot
        tot += 1
        ok += bool(cond)
        print(f"  {'OK ' if cond else '!! '} [{tot}] {label}")

    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    chain_c = np.eye(4)
    chain_c[:3, :3] = R
    chain_c[:3, 3] = [0.11, -0.22, 0.33]
    p = np.array([0.40, 0.02, 0.035])

    T = align_transform(chain_c[:3, 3], p, 0.0)
    out = T @ chain_c
    check("yaw 0: 파지점이 정확히 p 로 간다",
          np.allclose(out[:3, 3], p, atol=1e-12))
    check("yaw 0: 자세가 그대로다", np.allclose(out[:3, :3], R, atol=1e-12))

    yaw = np.deg2rad(37.0)
    out = align_transform(chain_c[:3, 3], p, yaw) @ chain_c
    check("yaw 37도: 위치는 여전히 p",
          np.allclose(out[:3, 3], p, atol=1e-12))
    check("yaw 37도: 자세가 Rz(37) @ R",
          np.allclose(out[:3, :3], rz(yaw)[:3, :3] @ R, atol=1e-12))

    # Discriminating row: the wrong composition order must NOT land on p.
    # 판별력 행: 곱 순서를 바꾸면 p 에 안 가야 한다. 가면 이 검정은 힘이 없다.
    tp, tg = np.eye(4), np.eye(4)
    tp[:3, 3] = p
    tg[:3, 3] = -chain_c[:3, 3]
    wrong = (tg @ rz(yaw) @ tp) @ chain_c
    check("판별력: 곱 순서를 뒤집으면 p 에 안 간다",
          not np.allclose(wrong[:3, 3], p, atol=1e-6))

    # Discriminating row: yaw must actually change the pose.
    # 판별력 행: yaw 가 자세를 실제로 바꿔야 한다.
    check("판별력: yaw 37도와 yaw 0 의 자세가 다르다",
          not np.allclose((align_transform(chain_c[:3, 3], p, yaw) @ chain_c)[:3, :3],
                          (align_transform(chain_c[:3, 3], p, 0.0) @ chain_c)[:3, :3]))

    g = grid(3, 3, 2, 4)
    check("격자 크기가 모수와 맞는다 (3*3*2*4 = 72)", len(g) == 72)
    check("격자가 전부 포락선 안이다",
          all(ENVELOPE["x"][0] - 1e-9 <= q[0][0] <= ENVELOPE["x"][1] + 1e-9
              and ENVELOPE["y"][0] - 1e-9 <= q[0][1] <= ENVELOPE["y"][1] + 1e-9
              and ENVELOPE["z"][0] - 1e-9 <= q[0][2] <= ENVELOPE["z"][1] + 1e-9
              for q in g))

    print(f"자체검증 {ok} / {tot}")
    return 0 if ok == tot else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", help="v10 디렉터리")
    ap.add_argument("--task", help="시뮬 task yaml")
    ap.add_argument("--reference-demos", help="속도 예산 실측용 시뮬 시연 디렉터리")
    ap.add_argument("--out", default="grasp_anchor.json")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--pre", type=int, default=2)
    ap.add_argument("--post", type=int, default=6)
    ap.add_argument("--ik-tol", type=float, default=0.008)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N편만 (0=전부)")
    ap.add_argument("--nx", type=int, default=5)
    ap.add_argument("--ny", type=int, default=5)
    ap.add_argument("--nz", type=int, default=3)
    ap.add_argument("--nyaw", type=int, default=8)
    ap.add_argument("--top", type=int, default=5, help="2단으로 넘길 후보 수")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    for need in ("dataset", "task", "reference_demos"):
        if not getattr(a, need):
            raise SystemExit(f"!! --{need.replace('_','-')} 가 필요하다 (또는 --selftest)")

    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 배치값을 내지 않는다")
    print()

    eps, dropped = load_episodes(Path(a.dataset).expanduser(), a.horizon,
                                 a.pre, a.post, a.limit)
    print(f"에피소드 사용 {len(eps)} / 읽음 {len(eps) + len(dropped)}")
    for d in dropped[:5]:
        print(f"   버림 {d['episode']}: {d['reason']}")
    if not eps:
        raise SystemExit("!! 쓸 편이 0개다. 범위가 비었는지 먼저 본다")

    limits = measure_reference_limits(Path(a.reference_demos).expanduser())
    print(f"속도 예산 실측 {limits['episodes_used']} / {limits['episodes_found']}편")

    from simulation.env import PickEnv
    env = PickEnv(task_path=str(Path(a.task).expanduser()))
    env.reset(0)
    T_home = env.tcp().copy()
    print("시뮬 홈 EEF 위치", np.round(T_home[:3, 3], 4))

    # 기준선 — 지금 하는 방식(시뮬 홈에 시작을 박는다)
    base_ok = sum(1 for e in eps
                  if check_episode(env, e["chain"], T_home, limits, a.horizon,
                                   a.ik_tol, e["dts"], e["span"])["episode_ok"])
    print(f"\n기준선(시뮬 홈 정렬): 파지구간 전부 통과 {base_ok} / {len(eps)}편\n")

    cands = grid(a.nx, a.ny, a.nz, a.nyaw)
    print(f"1단 — 후보 {len(cands)}개 × {len(eps)}편 = {len(cands)*len(eps)} IK")
    stage1 = []
    for i, (p, yaw) in enumerate(cands):
        n = sum(1 for e in eps
                if closure_reachable(env, e,
                                     align_transform(e["chain"][e["closure"]][:3, 3], p, yaw),
                                     a.ik_tol))
        stage1.append({"p": p.tolist(), "yaw_deg": round(np.rad2deg(yaw), 2), "closure_ok": n})
        if (i + 1) % 50 == 0:
            print(f"   {i+1} / {len(cands)}")
    stage1.sort(key=lambda r: -r["closure_ok"])
    print(f"1단 최고 닫힘통과 {stage1[0]['closure_ok']} / {len(eps)}편")

    print(f"\n2단 — 상위 {a.top}개 후보에 파지 구간 전체")
    stage2 = []
    for r in stage1[:a.top]:
        p, yaw = np.array(r["p"]), np.deg2rad(r["yaw_deg"])
        n_ok = n_way = n_way_ok = 0
        for e in eps:
            res = check_episode(env, e["chain"],
                                align_transform(e["chain"][e["closure"]][:3, 3], p, yaw),
                                limits, a.horizon, a.ik_tol, e["dts"], e["span"])
            n_ok += int(res["episode_ok"])
            n_way += res["waypoints"]
            n_way_ok += res["waypoints_ok"]
        row = {**r, "episodes_ok": n_ok, "episodes": len(eps),
               "waypoints_ok": n_way_ok, "waypoints": n_way}
        stage2.append(row)
        print(f"   p={np.round(p,3)} yaw={r['yaw_deg']:>7.2f}도  "
              f"편 {n_ok}/{len(eps)} · 웨이포인트 {n_way_ok}/{n_way}")
    stage2.sort(key=lambda r: (-r["episodes_ok"], -r["waypoints_ok"]))

    best = stage2[0]
    out = {
        "_WHAT_THIS_IS": "우리가 내는 배치 제안값이다. 실물 베이스 캘리브레이션이 아니다",
        "_CONFIRM_WITH": "황도경(정렬 좌표) · 김현석(실제로 그 자리에 놓을 수 있는지)",
        "_criterion": f"파지 순간 -{a.pre} ~ +{a.post} 행만 판정. 시연 전체 궤적이 아니다",
        "_envelope_source": "MEASURE_task2_envelope_0916 측면파지 실측 (handoff SO-101, base_height 0)",
        "episodes_used": len(eps),
        "episodes_dropped": dropped,
        "baseline_home_alignment_ok": base_ok,
        "grid": {"nx": a.nx, "ny": a.ny, "nz": a.nz, "nyaw": a.nyaw, "candidates": len(cands)},
        "best": best,
        "stage2": stage2,
        "stage1_top20": stage1[:20],
    }
    Path(a.out).expanduser().write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"\n제안 배치  p = {np.round(np.array(best['p']), 4)} m · yaw = {best['yaw_deg']}도")
    print(f"  파지구간 통과  {best['episodes_ok']} / {best['episodes']}편   "
          f"(기준선 {base_ok} / {len(eps)})")
    print(f"  웨이포인트     {best['waypoints_ok']} / {best['waypoints']}")
    print(f"→ {a.out}")
    print("\n⚠️ 이 값은 제안이다. 실물에 올리기 전에 황도경·김현석 확인.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
