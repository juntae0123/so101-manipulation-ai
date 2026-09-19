"""Where should the object sit so the grasp is reachable? Solved from the demos.
대상물을 어디에 놓아야 파지가 도달권에 드는가. 시연에서 직접 푼다.

⚠️ 2026-09-19 전면 수정 — 초판은 틀렸다
--------------------------------------
초판은 **시연 자체의 파지 자세를 유지한 채 위치와 월드 yaw 만** 흔들었다.
파일럿에서 후보 72개 × 10편 = 720 IK 가 **전부 0** 이었다.

왜 틀렸나: v10 은 `robot_base_alignment_applied: false` 다. 시연 궤적의 자세는
**로봇 좌표계에 놓인 적이 없다.** 월드 z 둘레 회전만으로는 정렬되지 않는 자세를
아무리 옮겨봐야 0 이 나온다. 그리고 이건 이미 답이 있던 문제였다 —
`MEASURE_real_grasp_ik_0917` 이 **앵커를 시뮬 파지 pose 에 박으면 533/535 = 99.6%**
라고 적어놨다. 나는 그 문서를 읽고도 다른 앵커를 새로 만들었다.

지금 판: 0917 과 **같은 앵커식**을 쓰고, 자유 변수는 **물체를 놓는 자리** 하나다.

    T_grasp_sim = 물체 위치로부터 전문가 식 그대로 계산 (접근 방향이 자세를 정한다)
    T_align     = T_grasp_sim @ inv(chain[closure])
    T_base[t]   = T_align @ chain[t]

즉 묻는 것은 *"물체를 (x, y) 에 놓으면 파지 순간이 도달권에 드는가"* 다.
자세는 자유 변수가 아니다 — 물체 위치가 접근 방향을 정하고 접근 방향이 자세를 정한다.

양성 대조 (없으면 아무 결론도 안 낸다)
--------------------------------------
시드 0 의 **실제 물체 위치**를 후보에 반드시 넣고, 그 조건이 통과하는지 먼저 본다.
그건 0917 과 정확히 같은 설정이므로 높게 나와야 한다. **0 이면 도구가 틀린 것이고,
그 상태에서 "어디에 놔도 안 된다" 는 결론을 내지 않는다.**
초판에는 이 행이 없었다. 그래서 0/720 을 놓고 데이터 탓을 할 뻔했다.

무엇을 성공으로 보나
--------------------
**파지 순간 −pre ~ +post 행만.** 시연 전체 궤적이 아니다.

Usage
-----
  # [서버]
  ~/envs/handoff312/bin/python AI/tools/solve_grasp_anchor.py --selftest
  cd ~/handoff && ~/envs/handoff312/bin/python ~/S15P21A103/AI/tools/solve_grasp_anchor.py --dataset <v10> --task configs/can_side.yaml --reference-demos ~/handoff/outputs/demos_6000 --out ~/handoff/outputs/grasp_anchor.json

되돌리기
--------
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

# 측면 파지 도달 포락선 실측 (handoff SO-101, base_height 0) — MEASURE_task2_envelope_0916
ENVELOPE_X = (0.34, 0.46)
ENVELOPE_Y = (-0.175, 0.175)


def grasp_pose_for_object(obj_xyz: np.ndarray) -> np.ndarray:
    """The sim expert's side-grasp TCP pose for an object at this position.
    이 위치에 물체가 있을 때 시뮬 전문가가 쓰는 측면 파지 TCP pose.

    `convert_v10_to_umi.sim_grasp_pose` 와 **같은 식**이다. 거기서 그대로 가져왔다.
    자세는 접근 방향 `ap` 가 정한다 — 즉 **물체 위치가 자세를 정한다.**
    자세를 따로 흔들 자유 변수로 두면 전문가가 실제로 내는 자세와 달라진다."""
    obj = np.asarray(obj_xyz, dtype=np.float64)
    ap = np.r_[obj[:2] - np.array([0.04, 0.0]), 0.0]
    n = np.linalg.norm(ap)
    if n < 1e-9:
        raise ValueError("물체가 접근 기준점과 같은 위치다")
    ap = ap / n
    T = np.eye(4)
    T[:3, :3] = np.column_stack([np.cross([0, 0, 1], ap), [0, 0, 1], ap])
    T[:3, 3] = obj + ap * 0.013
    return T


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


def object_grid(z_obj: float, nx: int, ny: int) -> list[np.ndarray]:
    """Candidate object positions. z comes from the object, not from us.
    후보 물체 위치. z 는 물체가 정한다. 우리가 고르는 값이 아니다."""
    return [np.array([x, y, z_obj])
            for x in np.linspace(*ENVELOPE_X, nx)
            for y in np.linspace(*ENVELOPE_Y, ny)]


def align_for(obj_xyz: np.ndarray, chain_closure: np.ndarray) -> np.ndarray:
    """0917 의 앵커식 그대로: T_align = T_grasp_sim @ inv(chain[closure])."""
    return grasp_pose_for_object(obj_xyz) @ np.linalg.inv(chain_closure)


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
    """Known-answer rows plus deliberately wrong inputs.
    정답 아는 행 + 고의로 틀린 입력."""
    ok = tot = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok, tot
        tot += 1
        ok += bool(cond)
        print(f"  {'OK ' if cond else '!! '} [{tot}] {label:<50} {detail}")

    obj = np.array([0.40, 0.00, 0.051])
    T = grasp_pose_for_object(obj)
    check("파지 pose 가 정규직교", np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12))
    check("파지 pose 가 오른손계", abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-12)
    check("TCP 가 물체보다 접근점 쪽으로 13mm",
          abs(np.linalg.norm(T[:3, 3] - obj) - 0.013) < 1e-12)
    check("파지 높이 = 물체 중심 높이", abs(T[2, 3] - obj[2]) < 1e-12)
    # 판별력: y 를 옮기면 접근 방향이 바뀌므로 자세가 바뀌어야 한다.
    T2 = grasp_pose_for_object(np.array([0.40, 0.10, 0.051]))
    check("판별력: 물체를 옮기면 자세도 바뀐다 (위치만이 아니다)",
          not np.allclose(T[:3, :3], T2[:3, :3], atol=1e-6))

    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    cc = np.eye(4)
    cc[:3, :3] = R
    cc[:3, 3] = [0.11, -0.22, 0.33]
    out = align_for(obj, cc) @ cc
    check("앵커식: 닫힘 pose 가 정확히 T_grasp_sim 이 된다 (정답 아는 행)",
          np.allclose(out, T, atol=1e-12))
    # 판별력: 역행렬을 빼면 안 맞아야 한다.
    check("판별력: inv(chain[closure]) 를 빼면 안 맞는다",
          not np.allclose(grasp_pose_for_object(obj) @ cc, T, atol=1e-6))

    g = object_grid(0.051, 3, 3)
    check("격자 크기가 모수와 맞는다 (3*3)", len(g) == 9)
    check("격자 z 가 전부 물체 높이", all(abs(p[2] - 0.051) < 1e-12 for p in g))
    check("격자가 포락선 안이다",
          all(ENVELOPE_X[0] - 1e-9 <= p[0] <= ENVELOPE_X[1] + 1e-9
              and ENVELOPE_Y[0] - 1e-9 <= p[1] <= ENVELOPE_Y[1] + 1e-9 for p in g))
    try:
        grasp_pose_for_object(np.array([0.04, 0.0, 0.05]))
        died = False
    except ValueError:
        died = True
    check("판별력: 접근점과 같은 위치면 죽는다", died)

    print(f"자체검증 {ok} / {tot}")
    return 0 if ok == tot else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset")
    ap.add_argument("--task")
    ap.add_argument("--reference-demos")
    ap.add_argument("--out", default="grasp_anchor.json")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--pre", type=int, default=2)
    ap.add_argument("--post", type=int, default=6)
    ap.add_argument("--ik-tol", type=float, default=0.008)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nx", type=int, default=7)
    ap.add_argument("--ny", type=int, default=7)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--control-seed", type=int, default=0,
                    help="양성 대조로 쓸 시뮬 시드 (그 시드의 실제 물체 위치)")
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
    if not eps:
        raise SystemExit("!! 쓸 편이 0개다")

    limits = measure_reference_limits(Path(a.reference_demos).expanduser())
    print(f"속도 예산 실측 {limits['episodes_used']} / {limits['episodes_found']}편")

    from simulation.env import PickEnv
    env = PickEnv(task_path=str(Path(a.task).expanduser()))
    env.reset(a.control_seed)
    obj0 = env.data.body("object").xpos.copy()
    print(f"양성 대조 물체 위치 (시드 {a.control_seed}) {np.round(obj0, 4)}")

    def score(obj_xyz: np.ndarray) -> dict:
        n_ok = n_way = n_way_ok = 0
        why: dict[str, int] = {}
        for e in eps:
            r = check_episode(env, e["chain"], align_for(obj_xyz, e["chain"][e["closure"]]),
                              limits, a.horizon, a.ik_tol, e["dts"], e["span"])
            n_ok += int(r["episode_ok"])
            n_way += r["waypoints"]
            n_way_ok += r["waypoints_ok"]
            for w in r["reasons"]:
                if w:
                    why[w.split(">")[0]] = why.get(w.split(">")[0], 0) + 1
        return {"episodes_ok": n_ok, "episodes": len(eps),
                "waypoints_ok": n_way_ok, "waypoints": n_way, "reject_reasons": why}

    # ── 양성 대조 먼저. 여기서 0 이면 아무 결론도 내지 않는다.
    ctrl = score(obj0)
    print(f"\n=== 양성 대조 (0917 과 같은 설정) ===")
    print(f"  편 {ctrl['episodes_ok']} / {ctrl['episodes']} · "
          f"웨이포인트 {ctrl['waypoints_ok']} / {ctrl['waypoints']}")
    if ctrl["reject_reasons"]:
        print(f"  거부 사유 {ctrl['reject_reasons']}")
    if ctrl["waypoints_ok"] == 0:
        raise SystemExit(
            "!! 양성 대조가 0 이다. 0917 은 같은 앵커로 533/535 를 냈다.\n"
            "   데이터가 아니라 이 도구(또는 env·계약)가 틀린 것이다.\n"
            "   '어디에 놔도 안 된다' 는 결론을 내지 않는다. 중단")

    grid = object_grid(float(obj0[2]), a.nx, a.ny)
    print(f"\n1단 — 물체 위치 후보 {len(grid)}개 × {len(eps)}편 = {len(grid)*len(eps)} IK")
    stage1 = []
    for i, p in enumerate(grid):
        n = sum(1 for e in eps
                if closure_reachable(env, e, align_for(p, e["chain"][e["closure"]]), a.ik_tol))
        stage1.append({"obj": p.tolist(), "closure_ok": n})
        if (i + 1) % 10 == 0:
            print(f"   {i+1} / {len(grid)}  (최고 {max(r['closure_ok'] for r in stage1)})")
    stage1.sort(key=lambda r: -r["closure_ok"])
    print(f"1단 최고 닫힘통과 {stage1[0]['closure_ok']} / {len(eps)}편")

    print(f"\n2단 — 상위 {a.top}개에 파지 구간 전체")
    stage2 = []
    for r in stage1[:a.top]:
        s = score(np.array(r["obj"]))
        stage2.append({**r, **s})
        print(f"   물체 {np.round(np.array(r['obj']),3)}  "
              f"편 {s['episodes_ok']}/{s['episodes']} · "
              f"웨이포인트 {s['waypoints_ok']}/{s['waypoints']}")
    stage2.sort(key=lambda r: (-r["episodes_ok"], -r["waypoints_ok"]))
    best = stage2[0]

    out = {
        "_WHAT_THIS_IS": "물체를 놓을 자리 제안값이다. 실물 베이스 캘리브레이션이 아니다",
        "_CONFIRM_WITH": "황도경(정렬 좌표) · 김현석(그 자리에 실제로 놓을 수 있는지)",
        "_anchor": "T_align = T_grasp_sim(물체위치) @ inv(chain[closure]) — 0917 과 동일",
        "_criterion": f"파지 순간 -{a.pre} ~ +{a.post} 행만. 시연 전체 궤적이 아니다",
        "_positive_control": ctrl,
        "episodes_used": len(eps), "episodes_dropped": dropped,
        "grid": {"nx": a.nx, "ny": a.ny, "candidates": len(grid)},
        "best": best, "stage2": stage2, "stage1_top20": stage1[:20],
    }
    Path(a.out).expanduser().write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"\n제안 물체 위치  {np.round(np.array(best['obj']), 4)} m")
    print(f"  파지구간 통과 {best['episodes_ok']} / {best['episodes']}편  "
          f"(양성 대조 {ctrl['episodes_ok']} / {ctrl['episodes']})")
    print(f"→ {a.out}")
    print("\n⚠️ 제안값이다. 실물에 올리기 전에 황도경·김현석 확인.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
