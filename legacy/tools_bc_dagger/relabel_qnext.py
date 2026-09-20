"""Experimental ctrl -> q[t+1] relabeler. Never writes into the source dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from contract.episode import read_episode, validate, write_dataset_index


def shifted(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    state = arrays["state"]
    if state.ndim != 2 or state.shape[0] < 2:
        raise ValueError(f"state must contain at least 2 ticks, got {state.shape}")

    n = state.shape[0]
    result: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key == "action":
            result[key] = state[1:].copy()
        elif key == "action_timestamp":
            result[key] = arrays["state_timestamp"][:-1].copy()
        elif value.ndim >= 1 and value.shape[0] == n:
            result[key] = value[:-1].copy()
        else:
            result[key] = value.copy()
    return result


def self_test() -> None:
    state = np.arange(24, dtype=np.float32).reshape(4, 6)
    arrays = {
        "state": state,
        "action": np.full((4, 6), -99, dtype=np.float32),
        "state_timestamp": np.arange(4, dtype=np.float64) / 30,
        "action_timestamp": np.arange(4, dtype=np.float64) / 30,
        "image__cam": np.arange(12, dtype=np.uint8).reshape(4, 3),
    }
    out = shifted(arrays)
    assert out["state"].shape == (3, 6)
    assert np.array_equal(out["state"], state[:-1])
    assert np.array_equal(out["action"], state[1:])
    assert np.array_equal(out["action_timestamp"], arrays["state_timestamp"][:-1])
    assert out["image__cam"].shape[0] == 3
    assert np.all(arrays["action"] == -99)
    print("fixture: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    self_test()

    if args.destination.exists():
        raise FileExistsError(f"destination already exists: {args.destination}")
    args.destination.mkdir(parents=True)

    paths = sorted(args.source.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episodes in {args.source}")

    for src_npz in paths:
        with np.load(src_npz) as z:
            arrays = {key: z[key] for key in z.files}
        converted = shifted(arrays)

        dst_npz = args.destination / src_npz.name
        np.savez_compressed(dst_npz, **converted)

        src_json = src_npz.with_suffix(".json")
        metadata = json.loads(src_json.read_text(encoding="utf-8"))
        metadata["n_steps"] = int(converted["state"].shape[0])
        notes = dict(metadata.get("notes") or {})
        notes.update({
            "experimental_only": True,
            "action_semantics": "q[t+1]",
            "source_dataset": str(args.source),
            "terminal_rule": "drop_last_observation",
        })
        metadata["notes"] = notes
        dst_npz.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    write_dataset_index(
        args.destination,
        extra={
            "experimental_only": True,
            "action_semantics": "q[t+1]",
            "source_dataset": str(args.source),
            "terminal_rule": "drop_last_observation",
        },
    )

    violations = 0
    for path in sorted(args.destination.glob("*.npz")):
        problems = validate(read_episode(path))
        if problems:
            violations += 1
            print(path.name, problems)
    if violations:
        raise RuntimeError(f"contract violations: {violations}")

    marker = args.destination / "EXPERIMENT_ONLY_DO_NOT_MERGE.txt"
    marker.write_text(
        "Temporary q[t+1] A/B artifact. Not a canonical dataset.\n",
        encoding="utf-8",
    )
    print(f"converted={len(paths)}, violations=0, destination={args.destination}")


if __name__ == "__main__":
    main()
