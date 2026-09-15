from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import runtime_limits

runtime_limits.claim("collect_dagger_segments")
runtime_limits.torch_threads()

import numpy as np

from contract.episode import (
    ACTION_DIM,
    CONTRACT_VERSION,
    RANGE_TOLERANCE,
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
from sim.mujoco.build_scene import DEFAULT_CONFIG
from sim.mujoco.env import MujocoPickEnv
from tracking.exp_log import _git_rev, file_digest, log_run


def snapshot(env: MujocoPickEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    return (
        env.data.qpos.copy(),
        env.data.qvel.copy(),
        env.data.ctrl.copy(),
        float(env.data.time),
    )


def assert_unchanged(
    before: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    env: MujocoPickEnv,
) -> None:
    after = snapshot(env)
    for name, a, b in zip(("qpos", "qvel", "ctrl"), before[:3], after[:3]):
        if not np.array_equal(a, b):
            raise RuntimeError(f"expert query mutated {name}")
    if before[3] != after[3]:
        raise RuntimeError("expert query mutated time")


def range_excess(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if not np.isfinite(arr).all():
        return float("inf")
    return max(
        float(np.max(arr - 1.0)),
        float(np.max(-1.0 - arr)),
        0.0,
    )


def fresh(cameras: list[str]) -> dict:
    return {
        "images": {camera: [] for camera in cameras},
        "state": [],
        "action": [],
        "timestamp": [],
    }


def close_segment(current: dict, segments: list[dict], stats: Counter) -> dict:
    n = len(current["state"])
    if n >= 2:
        segments.append(current)
    else:
        stats["short_valid_ticks_dropped"] += n
    return fresh(list(current["images"]))


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

    policies = [load_policy(path, device="cpu") for path in args.policy_ckpt]
    stats: Counter = Counter()
    phase_invalid: Counter = Counter()
    max_state_excess = 0.0
    max_action_excess = 0.0
    segment_index = 0

    with MujocoPickEnv(
        render=True,
        object_jitter_m=0.05,
        max_ticks=200,
    ) as env:
        cameras = env.camera_names
        expert = ScriptedFeedbackPolicy(env)

        for episode_index in range(args.episodes):
            policy_index = episode_index % len(policies)
            policy = policies[policy_index]
            seed = args.seed_base + episode_index

            obs = env.reset(seed=seed)
            policy.reset(seed)
            expert.reset(seed)

            current = fresh(cameras)
            segments: list[dict] = []
            behavior_success = False

            for tick in range(env.max_ticks):
                behavior_action = policy.act(obs)

                if tick >= args.label_start_tick:
                    stats["queries"] += 1
                    before = snapshot(env)
                    failures_before = expert.ik_failures
                    label = expert.act(obs)
                    assert_unchanged(before, env)
                    stats["nonmutation_fixture_passes"] += 1

                    ik_invalid = expert.ik_failures > failures_before
                    state_excess = range_excess(obs.state)
                    action_excess = range_excess(label)
                    range_invalid = (
                        state_excess > RANGE_TOLERANCE
                        or action_excess > RANGE_TOLERANCE
                    )

                    max_state_excess = max(max_state_excess, state_excess)
                    max_action_excess = max(max_action_excess, action_excess)

                    if ik_invalid or range_invalid:
                        if ik_invalid:
                            stats["invalid_ik_ticks"] += 1
                            phase_invalid[expert.phase] += 1
                        if range_invalid:
                            stats["invalid_range_ticks"] += 1
                        if ik_invalid and range_invalid:
                            stats["invalid_both_ticks"] += 1
                        current = close_segment(current, segments, stats)
                    else:
                        stats["valid_ticks"] += 1
                        for camera in cameras:
                            current["images"][camera].append(
                                obs.images[camera].copy()
                            )
                        current["state"].append(
                            np.asarray(obs.state, dtype=np.float32).copy()
                        )
                        current["action"].append(
                            np.asarray(label, dtype=np.float32).copy()
                        )
                        current["timestamp"].append(float(obs.timestamp))

                obs = env.step(behavior_action)
                behavior_success = behavior_success or env.is_success()

            current = close_segment(current, segments, stats)
            stats["behavior_successes"] += int(behavior_success)

            for local_segment_index, segment in enumerate(segments):
                n = len(segment["state"])
                # 계약 0.3.0 은 action[t] = state[t+1] 을 강제한다 (validate).
                # 그런데 전문가가 내린 명령(label)은 그것과 다른 값이고, **학습에
                # 쓸모 있는 것은 그쪽이다** — 0914 확정 🟢: state[t+1] 타깃은
                # 0/300, ctrl 타깃은 22/300. 그래서 계약 action 은 규약대로 만들고
                # 명령은 사이드카로 따로 남긴다. 계약 npz 는 건드리지 않는다.
                state_arr = np.stack(segment["state"]).astype(np.float32)
                command_arr = np.stack(segment["action"]).astype(np.float32)
                action_arr = np.vstack(
                    [state_arr[1:], state_arr[-1:]]
                ).astype(np.float32)
                episode_id = (
                    f"dagger_seg_{episode_index:05d}_{local_segment_index:03d}"
                )
                timestamps = np.asarray(
                    segment["timestamp"], dtype=np.float64
                )
                ep = Episode(
                    meta=EpisodeMeta(
                        episode_id=episode_id,
                        skill_id="pick_place",
                        task="pick_cube_2cm_dagger_segment",
                        source="sim",
                        success=behavior_success,
                        n_steps=n,
                        control_rate_hz=env.control_rate_hz,
                        cameras=list(cameras),
                        contract_version=CONTRACT_VERSION,
                        collected_by="김준태(트랙B)",
                        config_sha=file_digest(DEFAULT_CONFIG),
                        git_rev=_git_rev(),
                        notes={
                            "dagger_round": "1b",
                            "behavior_checkpoint": str(
                                args.policy_ckpt[policy_index]
                            ),
                            "behavior_seed": seed,
                            "source_episode_index": episode_index,
                            "segment_index": local_segment_index,
                            "label_policy": "ScriptedFeedbackPolicy",
                            "label_start_tick": args.label_start_tick,
                            "expert_action_executed": False,
                        },
                    ),
                    images={
                        camera: np.stack(frames).astype(np.uint8)
                        for camera, frames in segment["images"].items()
                    },
                    state=state_arr,
                    state_timestamp=timestamps,
                    action=action_arr,
                    action_timestamp=timestamps.copy(),
                )
                problems = validate(ep)
                if problems:
                    raise RuntimeError(
                        f"{episode_id} contract violation: {problems}"
                    )
                path = write_episode(ep, args.out)
                # 사이드카: 전문가가 내린 명령. `read_episode` 도 `validate` 도
                # 이 파일을 보지 않으므로 계약은 그대로다.
                # 학습은 `train_bc.py --target-sidecar command` 로 집어 쓴다.
                np.save(path.with_suffix(".command.npy"), command_arr)
                stats["stored_ticks"] += n
                stats["segments"] += 1
                segment_index += 1

            if (episode_index + 1) % 10 == 0:
                print(
                    f"[{episode_index + 1:3d}/{args.episodes}] "
                    f"queries={stats['queries']} "
                    f"valid={stats['valid_ticks']} "
                    f"stored={stats['stored_ticks']} "
                    f"segments={stats['segments']}",
                    flush=True,
                )

    expected_queries = (
        args.episodes * (200 - args.label_start_tick)
    )
    if stats["queries"] != expected_queries:
        raise RuntimeError(
            f"query count {stats['queries']} != {expected_queries}"
        )
    if stats["nonmutation_fixture_passes"] != stats["queries"]:
        raise RuntimeError("nonmutation fixture count mismatch")

    contract_violations = 0
    for path in sorted(args.out.glob("*.npz")):
        contract_violations += len(validate(read_episode(path)))

    valid_rate = stats["valid_ticks"] / stats["queries"]
    stored_rate = stats["stored_ticks"] / stats["queries"]
    gates = {
        "valid_label_rate_ge_95pct": valid_rate >= 0.95,
        "stored_label_rate_ge_95pct": stored_rate >= 0.95,
        "contract_violations_zero": contract_violations == 0,
    }
    gates["train_go"] = all(gates.values())

    write_dataset_index(
        args.out,
        extra={
            "experimental_only": True,
            "dagger_round": "1b",
            "behavior_checkpoints": [
                str(path) for path in args.policy_ckpt
            ],
            "seed_base": args.seed_base,
            "episodes": args.episodes,
            "label_start_tick": args.label_start_tick,
        },
    )

    result = {
        "episodes": args.episodes,
        "queries": stats["queries"],
        "valid_ticks": stats["valid_ticks"],
        "stored_ticks": stats["stored_ticks"],
        "valid_label_rate": valid_rate,
        "stored_label_rate": stored_rate,
        "invalid_ik_ticks": stats["invalid_ik_ticks"],
        "invalid_range_ticks": stats["invalid_range_ticks"],
        "invalid_both_ticks": stats["invalid_both_ticks"],
        "short_valid_ticks_dropped": stats["short_valid_ticks_dropped"],
        "segments": stats["segments"],
        "behavior_successes": stats["behavior_successes"],
        "phase_invalid": dict(phase_invalid),
        "max_state_excess": max_state_excess,
        "max_action_excess": max_action_excess,
        "contract_violations": contract_violations,
        "gates": gates,
    }
    result_path = args.out / "dagger_segments_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.log:
        log_run(
            experiment="dagger_r1b_segments",
            author="김준태(트랙B)",
            issue="S15P21A103-34",
            conditions={
                "behavior_checkpoints": [
                    str(path) for path in args.policy_ckpt
                ],
                "episodes": args.episodes,
                "seed_base": args.seed_base,
                "label_start_tick": args.label_start_tick,
                "render": True,
                "policy_device": "cpu",
                "jitter_m": 0.05,
            },
            result=result,
        )

    return 0 if gates["train_go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
