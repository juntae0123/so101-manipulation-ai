from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from contract.episode import (
    CONTRACT_VERSION,
    Episode,
    EpisodeMeta,
    read_episode,
    validate,
    write_dataset_index,
    write_episode,
)
from policy.baselines import ScriptedFeedbackPolicy
from policy.bc import BCPolicy
from policy.act import load_policy
from sim.mujoco.env import MujocoPickEnv
from sim.mujoco.build_scene import DEFAULT_CONFIG
from tracking.exp_log import _git_rev, file_digest, log_run


def unchanged(before: tuple, env: MujocoPickEnv) -> None:
    after = (
        env.data.qpos.copy(),
        env.data.qvel.copy(),
        env.data.ctrl.copy(),
        float(env.data.time),
    )
    for name, a, b in zip(("qpos", "qvel", "ctrl"), before[:3], after[:3]):
        if not np.array_equal(a, b):
            raise RuntimeError(f"expert query mutated MuJoCo {name}")
    if before[3] != after[3]:
        raise RuntimeError("expert query mutated MuJoCo time")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-ckpt", type=Path, action="append", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=4000)
    parser.add_argument("--label-start-tick", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)

    ckpts = list(args.policy_ckpt)
    policies = [load_policy(path, device="cpu") for path in ckpts]

    queries = 0
    valid_queries = 0
    retained = 0
    behavior_successes = 0
    range_skipped = 0
    range_excesses: list[float] = []
    phase_failures: Counter[str] = Counter()
    policy_counts: Counter[int] = Counter()
    fixture_queries = 0

    with MujocoPickEnv(
        render=True,
        object_jitter_m=0.05,
        max_ticks=200,
    ) as env:
        expert = ScriptedFeedbackPolicy(env)
        cameras = env.camera_names

        for episode_index in range(args.episodes):
            policy_index = episode_index % len(policies)
            policy = policies[policy_index]
            policy_counts[policy_index] += 1
            seed = args.seed_base + episode_index

            obs = env.reset(seed=seed)
            policy.reset(seed)
            expert.reset(seed)

            images = {camera: [] for camera in cameras}
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            timestamps: list[float] = []
            episode_invalid = False
            episode_success = False

            for tick in range(env.max_ticks):
                behavior_action = policy.act(obs)

                if tick >= args.label_start_tick:
                    before = (
                        env.data.qpos.copy(),
                        env.data.qvel.copy(),
                        env.data.ctrl.copy(),
                        float(env.data.time),
                    )
                    failures_before = expert.ik_failures
                    label = expert.act(obs)
                    unchanged(before, env)
                    fixture_queries += 1

                    failed = expert.ik_failures > failures_before
                    queries += 1
                    valid_queries += int(not failed)

                    if failed:
                        episode_invalid = True
                        phase_failures[expert.phase] += 1

                    for camera in cameras:
                        images[camera].append(obs.images[camera].copy())
                    states.append(obs.state.copy())
                    actions.append(label.copy())
                    timestamps.append(float(obs.timestamp))

                obs = env.step(behavior_action)
                episode_success = episode_success or env.is_success()

            behavior_successes += int(episode_success)

            # 중간 tick 제거는 계약의 30Hz 연속 timestamp를 깨므로,
            # IK 실패가 있으면 해당 에피소드 전체를 제외한다.
            if episode_invalid:
                print(
                    f"[{episode_index + 1:3d}/{args.episodes}] DROP "
                    f"seed={seed} policy=seed{policy_index} expert IK failure",
                    flush=True,
                )
                continue

            n = len(states)
            meta = EpisodeMeta(
                episode_id=f"dagger_r1_{episode_index:05d}",
                skill_id="pick_place",
                task="pick_cube_2cm_dagger_r1",
                source="sim",
                success=episode_success,
                n_steps=n,
                control_rate_hz=env.control_rate_hz,
                cameras=cameras,
                contract_version=CONTRACT_VERSION,
                collected_by="김준태(트랙B)",
                config_sha=file_digest(DEFAULT_CONFIG),
                git_rev=_git_rev(),
                notes={
                    "dagger_round": 1,
                    "behavior_checkpoint": str(ckpts[policy_index]),
                    "behavior_seed": seed,
                    "label_policy": "ScriptedFeedbackPolicy",
                    "label_start_tick": args.label_start_tick,
                    "expert_action_executed": False,
                    "ik_failure_episode": False,
                },
            )
            ep = Episode(
                meta=meta,
                images={
                    camera: np.stack(frames).astype(np.uint8)
                    for camera, frames in images.items()
                },
                state=np.stack(states).astype(np.float32),
                state_timestamp=np.asarray(timestamps, dtype=np.float64),
                action=np.stack(actions).astype(np.float32),
                action_timestamp=np.asarray(timestamps, dtype=np.float64),
            )
            problems = validate(ep)
            if problems:
                state_range_only = all(
                    problem.startswith("state out of ")
                    for problem in problems
                )
                if state_range_only:
                    excess = max(
                        float(np.max(ep.state - 1.0)),
                        float(np.max(-1.0 - ep.state)),
                        0.0,
                    )
                    range_skipped += 1
                    range_excesses.append(excess)
                    print(
                        f"[{episode_index + 1:3d}/{args.episodes}] DROP_RANGE "
                        f"seed={seed} excess_norm={excess:.6g}",
                        flush=True,
                    )
                    continue
                raise RuntimeError(
                    f"contract violation episode {episode_index}: {problems}"
                )
            write_episode(ep, args.out)
            retained += 1
            print(
                f"[{episode_index + 1:3d}/{args.episodes}] KEEP "
                f"seed={seed} policy=seed{policy_index} labels={n}",
                flush=True,
            )

    index = write_dataset_index(
        args.out,
        extra={
            "experimental_only": True,
            "dagger_round": 1,
            "behavior_checkpoints": [str(p) for p in ckpts],
            "seed_base": args.seed_base,
            "episodes_attempted": args.episodes,
            "label_start_tick": args.label_start_tick,
            "expert_action_executed": False,
        },
    )

    contract_violations = 0
    for path in sorted(args.out.glob("*.npz")):
        contract_violations += len(validate(read_episode(path)))

    valid_rate = valid_queries / queries
    retained_rate = retained / args.episodes
    range_skip_rate = range_skipped / args.episodes
    gates = {
        "valid_label_rate_ge_95pct": valid_rate >= 0.95,
        "retained_episode_rate_ge_80pct": retained_rate >= 0.80,
        "range_skip_rate_le_20pct": range_skip_rate <= 0.20,
        "contract_violations_zero": contract_violations == 0,
    }
    gates["train_go"] = all(gates.values())

    result = {
        "episodes_attempted": args.episodes,
        "episodes_retained": retained,
        "retained_episode_rate": retained_rate,
        "queries": queries,
        "valid_queries": valid_queries,
        "valid_label_rate": valid_rate,
        "fixture_nonmutation_queries": fixture_queries,
        "behavior_successes": behavior_successes,
        "range_skipped": range_skipped,
        "range_skip_rate": range_skip_rate,
        "range_excess_max": max(range_excesses) if range_excesses else 0.0,
        "range_excess_median": (
            float(np.median(range_excesses)) if range_excesses else 0.0
        ),
        "policy_episode_counts": dict(policy_counts),
        "phase_failures": dict(phase_failures),
        "contract_violations": contract_violations,
        "dataset_index": str(index),
        "gates": gates,
    }
    (args.out / "dagger_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out / "EXPERIMENT_ONLY_DO_NOT_MERGE.txt").write_text(
        "Temporary DAgger round-1 labels. Not a canonical dataset.\n",
        encoding="utf-8",
    )

    print("\n" + json.dumps(result, ensure_ascii=False, indent=2))

    if args.log:
        log_run(
            experiment="dagger_r1_collect",
            author="김준태(트랙B)",
            issue="S15P21A103-34",
            conditions={
                "behavior_checkpoints": [str(p) for p in ckpts],
                "episodes": args.episodes,
                "seed_base": args.seed_base,
                "label_start_tick": args.label_start_tick,
                "render": True,
                "policy_device": "cpu",
                "jitter_m": 0.05,
                "max_ticks": 200,
            },
            result=result,
        )

    return 0 if gates["train_go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
