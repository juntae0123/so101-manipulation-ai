"""Is the evaluation harness itself reproducible?
평가 하니스 자체가 재현되는가?

Measured 2026-09-03 🟢: the same checkpoint on the same seed block scored 11/100,
10/100 and 7/100 across three runs of the identical command, while `hold`, `zero`
and `scripted` -- none of which read images -- reproduced to the digit, failure
statistics included. That localises the cause to the image path but does not name
it. This tool names it, by probing the three links separately.
실측 2026-09-03 🟢: 같은 체크포인트를 같은 시드 블록으로 세 번 채점해 11/100,
10/100, 7/100 이 나왔다. 반면 이미지를 읽지 않는 `hold`·`zero`·`scripted` 는 실패
형태 통계까지 자리 하나까지 재현됐다. 원인이 이미지 경로에 있다는 것까지는 좁혀지지만
어디인지는 말해주지 않는다. 이 도구가 세 고리를 따로 찔러 그것을 지목한다.

  R 렌더    같은 상태를 두 번 렌더하면 바이트가 같은가
  P 추론    같은 관측을 두 번 넣으면 같은 행동이 나오는가
  E 롤아웃  같은 시드로 두 번 돌리면 같은 결과가 나오는가

Instrument-before-instrument: a harness whose numbers move on their own reports
improvements that are not there, and this project's only metric runs through it.
계측기의 계측기다. 스스로 움직이는 하니스는 없는 개선을 보고하고, 이 프로젝트의
유일한 지표가 그 하니스를 통과한다.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config
from sim.mujoco.env import MujocoPickEnv
from tracking.exp_log import file_digest, log_run


# Fixed before any number was produced. Bit-exact or it is not deterministic --
# no tolerance band, because a tolerance would hide exactly the effect measured.
# 어떤 수치도 나오기 전에 확정했다. 비트 단위로 같아야 한다. 허용 오차를 두지 않는다 —
# 오차를 두면 지금 재려는 효과가 정확히 그 안에 숨는다.
GATES: dict[str, str] = {
    "render_same_process": "한 프로세스에서 같은 상태를 두 번 렌더 → 픽셀 바이트 동일",
    "render_new_env": "새 env 인스턴스에서 같은 시드로 렌더 → 픽셀 바이트 동일",
    "policy_repeat": "같은 관측을 두 번 → 행동 비트 동일",
    "rollout_repeat": "같은 시드로 롤아웃 두 번 → 성공·틱·상승 동일",
}


def _hash(arrs: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(arrs):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrs[k]).tobytes())
    return h.hexdigest()[:16]


def probe_render(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Render the same scene twice, in one process and in a fresh env.
    같은 씬을 두 번 렌더한다 — 한 프로세스 안에서, 그리고 새 env 로."""
    with MujocoPickEnv(cfg, render=True, object_jitter_m=0.05) as env:
        o1 = env.reset(seed=seed)
        h1 = _hash(o1.images)
        # No step in between: the state is identical, so only the renderer can differ.
        # 사이에 step 이 없다. 상태가 동일하므로 다를 수 있는 것은 렌더러뿐이다.
        o2 = env._observe()
        h2 = _hash(o2.images)
        diff_same = {
            c: float(np.max(np.abs(o1.images[c].astype(np.int16) - o2.images[c].astype(np.int16))))
            for c in o1.images
        }
    with MujocoPickEnv(cfg, render=True, object_jitter_m=0.05) as env2:
        o3 = env2.reset(seed=seed)
        h3 = _hash(o3.images)
        diff_new = {
            c: float(np.max(np.abs(o1.images[c].astype(np.int16) - o3.images[c].astype(np.int16))))
            for c in o1.images
        }
    return {
        "hash_first": h1, "hash_same_process": h2, "hash_new_env": h3,
        "same_process_ok": h1 == h2, "new_env_ok": h1 == h3,
        "max_pixel_diff_same_process": diff_same,
        "max_pixel_diff_new_env": diff_new,
    }


def probe_policy(cfg: dict[str, Any], ckpt: Path, seed: int, device: str) -> dict[str, Any]:
    """Feed one fixed observation to the policy twice.
    고정된 관측 하나를 정책에 두 번 넣는다."""
    from policy.act import load_policy

    with MujocoPickEnv(cfg, render=True, object_jitter_m=0.05) as env:
        obs = env.reset(seed=seed)
    policy = load_policy(ckpt, device=device)
    a1 = policy.act(obs)
    a2 = policy.act(obs)
    return {
        "action_first": [round(float(v), 8) for v in a1],
        "max_abs_diff": float(np.max(np.abs(a1 - a2))),
        "ok": bool(np.array_equal(a1, a2)),
        "device": device,
    }


def probe_rollout(cfg: dict[str, Any], ckpt: Path, seed: int, device: str) -> dict[str, Any]:
    """Run the same seed twice through the full rollout.
    같은 시드로 전체 롤아웃을 두 번 돌린다."""
    from eval.rollout import rollout
    from policy.act import load_policy

    out = []
    with MujocoPickEnv(cfg, render=True, object_jitter_m=0.05) as env:
        policy = load_policy(ckpt, device=device)
        for _ in range(2):
            r = rollout(env, policy, seed)
            out.append({
                "success": r.success, "ticks": r.ticks, "lift_cm": round(r.lift_height_m * 100, 4),
                "min_pinch_xy_mm": r.min_pinch_xy_mm, "close_tick": r.close_tick,
            })
    return {"runs": out, "ok": out[0] == out[1]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-ckpt", type=Path, default=None,
                        help="없으면 렌더 검사만 한다 (P·E 절 생략)")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    print("게이트 기준 (결과 확인 전 확정):")
    for k, v in GATES.items():
        print(f"  [{k}] {v}")
    print("  허용 오차 없음 — 비트 단위로 같아야 통과한다\n")

    r = probe_render(cfg, args.seed)
    print("R 렌더")
    print(f"  한 프로세스 재렌더  {r['hash_first']} vs {r['hash_same_process']} → "
          f"{'동일' if r['same_process_ok'] else '**다름**'}  최대 픽셀차 {r['max_pixel_diff_same_process']}")
    print(f"  새 env 같은 시드    {r['hash_first']} vs {r['hash_new_env']} → "
          f"{'동일' if r['new_env_ok'] else '**다름**'}  최대 픽셀차 {r['max_pixel_diff_new_env']}")

    p_res: dict[str, Any] | None = None
    e_res: dict[str, Any] | None = None
    if args.policy_ckpt is not None:
        p_res = probe_policy(cfg, args.policy_ckpt, args.seed, args.device)
        print(f"\nP 추론 ({p_res['device']})")
        print(f"  같은 관측 두 번 → 최대 절대차 {p_res['max_abs_diff']:.3e} → "
              f"{'동일' if p_res['ok'] else '**다름**'}")

        e_res = probe_rollout(cfg, args.policy_ckpt, args.seed, args.device)
        print("\nE 롤아웃")
        for i, run in enumerate(e_res["runs"], 1):
            print(f"  {i}회  성공 {run['success']} · {run['ticks']}틱 · 상승 {run['lift_cm']}cm · "
                  f"최소거리 {run['min_pinch_xy_mm']}mm · 닫은틱 {run['close_tick']}")
        print(f"  → {'동일' if e_res['ok'] else '**다름**'}")
    else:
        print("\n⚠️ --policy-ckpt 가 없어 P·E 절을 건너뛴다. 렌더만으로는 원인을 좁힐 수 없다")

    print("\n판정:")
    if not r["same_process_ok"] or not r["new_env_ok"]:
        print("  원인 R — **렌더가 비결정적이다.** 같은 상태에서 다른 픽셀이 나온다.")
        print("  이미지를 쓰는 정책만 흔들린 관측(L37)과 일치한다.")
    elif p_res is not None and not p_res["ok"]:
        print("  원인 P — **추론이 비결정적이다.** 같은 입력에 다른 행동이 나온다.")
    elif e_res is not None and not e_res["ok"]:
        print("  원인 E — 렌더와 추론은 결정적인데 롤아웃 결과가 다르다.")
        print("  한 프로세스 안에서 재현되지 않는다면 물리 또는 누적 상태를 봐야 한다.")
    elif p_res is None:
        print("  R 통과. P·E 미측정이므로 원인 미확정.")
    else:
        offs = int(cfg["cameras"].get("offsamples", 4))
        print("  R·P·E 전부 통과 — **결정적이다.**")
        if offs == 0:
            print("  MSAA 꺼짐(offsamples=0). 실측 2026-09-03 🟢: offsamples=4 에서 R 1계조 다름 →")
            print("  0 에서 비트 동일. **MSAA 가 L37 의 원인이었다.** 수집·평가 모두 이 값을 유지한다.")
        else:
            print(f"  offsamples={offs} 인데 통과했다. 한 프로세스 안 결과이므로 별도 프로세스로")
            print("  두 번 돌려 대조해야 L37 을 닫을 수 있다.")

    if args.log:
        rec = log_run(
            experiment="check_determinism",
            author=args.author,
            issue="S15P21A103-35",
            conditions={
                "seed": args.seed, "device": args.device,
                "policy_ckpt": str(args.policy_ckpt) if args.policy_ckpt else None,
                "config_sha": file_digest(DEFAULT_CONFIG), "gates": GATES,
            },
            result={"render": r, "policy": p_res, "rollout": e_res},
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    ok = r["same_process_ok"] and r["new_env_ok"] and (p_res is None or p_res["ok"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
