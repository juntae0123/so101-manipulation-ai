"""Collect scripted demonstrations in simulation, in the data-contract format.
시뮬에서 스크립트 시연을 수집해 데이터 계약 포맷으로 저장한다.

Purpose is NOT to produce training data yet. It is to prove the contract can
actually be written and read back — a format that has never round-tripped is a
guess, not a contract. Real collection waits on S15P21A103-27 being agreed.
목적은 아직 학습 데이터 생산이 아니다. 계약 포맷이 실제로 쓰이고 다시 읽히는지
증명하는 것이다. 왕복해본 적 없는 포맷은 계약이 아니라 추측이다.
실제 수집은 S15P21A103-27 합의 이후다.

Staged collection is the rule for this project: 20 -> verify -> fix protocol ->
100 -> gate -> the rest. This script is the "20" stage tool.
이 프로젝트의 규칙은 단계식 수집이다: 20개 -> 검증 -> 프로토콜 확정 -> 100개
-> 게이트 -> 나머지. 이 스크립트는 그 "20개" 단계의 도구다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
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

from contract.ids import SKILL_IDS
from sim.mujoco.build_scene import DEFAULT_CONFIG, build_model, load_config, normalize
from tracking.exp_log import _git_rev, file_digest, log_run
from sim.mujoco.grasp import contact_on_object, gripper_geom_ids
from sim.mujoco.kinematics import pick_waypoints, solve_pose_ik


class EpisodeRecorder:
    """Capture one observation/action pair per control tick.
    제어 틱마다 관측/행동 한 쌍을 기록한다."""

    def __init__(self, model: mujoco.MjModel, cfg: dict, renderer: mujoco.Renderer) -> None:
        self.model = model
        self.cfg = cfg
        self.renderer = renderer
        from contract.episode import CAMERA_NAMES

        _all = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)
        ]
        # 계약 0.3.0 은 cam_wrist 한 대다 (D-AI-46). 씬은 2대 그대로 두고 기록만 거른다 —
        # configs/so101.yaml 은 양 트랙 공유 파일이라 건드리지 않는다.
        self.cameras = [c for c in _all if c in CAMERA_NAMES]
        if not self.cameras:
            raise RuntimeError(f"계약 카메라가 씬에 없다: {CAMERA_NAMES} vs {_all}")
        self.images: dict[str, list[np.ndarray]] = {c: [] for c in self.cameras}
        self.state: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.state_ts: list[float] = []
        self.action_ts: list[float] = []

    def capture(self, data: mujoco.MjData) -> None:
        """Record the current observation and the command that produced it.
        현재 관측과 그것을 만든 명령을 기록한다."""
        for cam in self.cameras:
            self.renderer.update_scene(data, camera=cam)
            frame = self.renderer.render()  # HWC uint8
            self.images[cam].append(np.transpose(frame, (2, 0, 1)).copy())  # -> CHW
        self.state.append(normalize(data.qpos[:6].copy(), self.cfg))
        self.action.append(normalize(data.ctrl[:6].copy(), self.cfg))
        # In sim both stamps are the same clock. On hardware they will not be —
        # the fields exist so S15P21A103-30 can measure the difference.
        # 시뮬에서는 두 타임스탬프가 같은 시계다. 실물은 아니다 —
        # S15P21A103-30이 그 차이를 계측할 수 있도록 필드를 둔다.
        self.state_ts.append(float(data.time))
        self.action_ts.append(float(data.time))

    def build(self, meta: EpisodeMeta) -> Episode:
        """Assemble the recorded buffers into a contract Episode.
        기록한 버퍼를 계약 Episode 로 조립한다."""
        meta.n_steps = len(self.state)
        meta.cameras = list(self.cameras)
        # 계약 0.3.0: action[t] = state[t+1]. 마지막 스텝은 자기 자신으로 채운다.
        # ⚠️ 이 규약 아래서는 기록해 둔 data.ctrl 이 버려진다 — 계약에 그 필드가 없다.
        state_arr = np.stack(self.state).astype(np.float32)
        action_arr = np.vstack([state_arr[1:], state_arr[-1:]]).astype(np.float32)
        # 내린 명령(ctrl)은 계약에 필드가 없어 버려진다. 가설 검증을 위해 노출한다.
        self.command = np.stack(self.action).astype(np.float32)
        return Episode(
            meta=meta,
            images={c: np.stack(v).astype(np.uint8) for c, v in self.images.items()},
            state=state_arr,
            state_timestamp=np.asarray(self.state_ts, dtype=np.float64),
            action=action_arr,
            action_timestamp=np.asarray(self.action_ts, dtype=np.float64),
        )


def _drive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rec: EpisodeRecorder,
    q_target: np.ndarray,
    gripper_cmd: float,
    seconds: float,
    rate_hz: float,
) -> None:
    """Interpolate to the target, capturing one frame per control tick.
    목표까지 보간하면서 제어 틱마다 한 프레임씩 기록한다."""
    q_start = data.ctrl[:5].copy()
    n_ctrl = max(1, int(seconds * rate_hz))
    sub = max(1, int((1.0 / rate_hz) / model.opt.timestep))
    for k in range(1, n_ctrl + 1):
        alpha = k / n_ctrl
        data.ctrl[:5] = (1 - alpha) * q_start + alpha * q_target[:5]
        data.ctrl[5] = gripper_cmd
        rec.capture(data)
        for _ in range(sub):
            mujoco.mj_step(model, data)


def collect_one(
    cfg: dict,
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    episode_id: str,
    object_xy: tuple[float, float],
    author: str,
    skill_id: str,
) -> Episode | None:
    """Collect one scripted pick episode, or None if the pose is unreachable.
    스크립트 파지 에피소드 하나를 수집한다. 자세가 도달 불가면 None."""
    g = cfg["grasp"]
    rate_hz = float(cfg["control"]["rate_hz"])
    offset = np.asarray(g["pinch_offset_local"], dtype=float)
    axis = np.asarray(g["approach_axis"], dtype=float)

    data = mujoco.MjData(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr] = object_xy[0]
    data.qpos[adr + 1] = object_xy[1]
    mujoco.mj_forward(model, data)
    data.ctrl[:] = data.qpos[:6]
    for _ in range(int(0.4 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    obj_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    jaws = gripper_geom_ids(model)
    obj_xyz = data.xpos[obj_bid].copy()
    z0 = float(obj_xyz[2])
    grasp_pt = obj_xyz + np.array([0.0, 0.0, float(g["grasp_z_offset_m"])])

    # One definition, shared with the evaluation baseline. See pick_waypoints.
    # 평가 baseline 과 공유하는 유일한 정의. pick_waypoints 참조.
    #
    # Required segments are solved up front, seed chained from one solution to
    # the next -- the same order the pre-refactor code used. Chaining turned out
    # not to matter for the rejection count (2 either way 🟢); what matters is
    # that the lift is not gated. See PickSegment.required.
    # 필수 구간은 앞에서 미리 풀고, 시드를 한 해에서 다음 해로 이어간다 —
    # 리팩터링 이전과 같은 순서다. 원인 분리 결과 시드 연결은 탈락 수와 무관했고
    # (양쪽 2개 🟢), 결정적인 것은 들어올리기에 게이트를 걸지 않는 것이다.
    # PickSegment.required 참조.
    segments = pick_waypoints(cfg, obj_xyz)
    q_seed = data.qpos[:6]
    q_pre: list[np.ndarray | None] = []
    for seg in segments:
        if seg.target is None or not seg.required:
            q_pre.append(None)          # 구동 시점에 결정한다
            continue
        res = solve_pose_ik(model, seg.target, offset, axis, q_init=q_seed)
        if not res.ok:
            return None
        q_pre.append(res.qpos)
        q_seed = res.qpos

    rec = EpisodeRecorder(model, cfg, renderer)
    for seg, pre in zip(segments, q_pre):
        if pre is not None:
            q_target = pre
        elif seg.target is None:
            q_target = data.ctrl[:5].copy()          # dwell · 제자리 닫기
        else:
            # Best effort: solved from the actual pose after closing, and a
            # failure here is tolerated -- as it was before the refactor.
            # 최선 노력: 닫은 뒤의 실제 자세에서 풀고, 실패해도 허용한다 —
            # 리팩터링 이전과 같다.
            q_target = solve_pose_ik(model, seg.target, offset, axis,
                                     q_init=data.qpos[:6]).qpos
        _drive(model, data, rec, q_target, seg.grip, seg.seconds, rate_hz)

    lifted = float(data.xpos[obj_bid][2]) - z0
    _, n_end = contact_on_object(model, data, obj_gid, jaws)
    success = lifted >= float(g["success_lift_m"]) and n_end > 0

    meta = EpisodeMeta(
        episode_id=episode_id,
        skill_id=skill_id,
        task="pick_cube_2cm",
        source="sim",
        success=success,
        n_steps=0,
        control_rate_hz=rate_hz,
        cameras=[],
        contract_version=CONTRACT_VERSION,
        collected_by=author,
        config_sha=file_digest(DEFAULT_CONFIG),
        git_rev=_git_rev(),
        notes={
            "object_init_xy": [round(float(object_xy[0]), 5), round(float(object_xy[1]), 5)],
            "lift_height_m": round(lifted, 5),
            "contacts_at_end": n_end,
            "simulator": "mujoco",
            "scripted": True,
            "caveat": "스크립트 시연이다. 사람 시연이 아니다. 궤적 다양성이 없다.",
        },
    )
    return rec.build(meta), rec.command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("datasets/sim_pick_v0"))
    parser.add_argument(
        "--skill-id",
        type=str,
        required=True,
        choices=list(SKILL_IDS),
        help="이 수집분이 어느 스킬의 시연인가. 계약 필수 필드다",
    )
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    model = build_model(cfg)
    width, height = cfg["cameras"]["resolution"]
    rng = np.random.default_rng(args.seed)
    base_xy = np.array(cfg["task"]["object"]["init_pos"][:2], dtype=float)

    written: list[Path] = []
    skipped = 0
    n_success = 0
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i in range(args.episodes):
            xy = base_xy + rng.uniform(-args.jitter, args.jitter, size=2)
            got = collect_one(
                cfg,
                model,
                renderer,
                f"ep_{i:05d}",
                (float(xy[0]), float(xy[1])),
                args.author,
                args.skill_id,
            )
            if got is None:
                skipped += 1
                print(f"[{i + 1:3d}/{args.episodes}] SKIP  도달 불가 xy=({xy[0]:+.3f},{xy[1]:+.3f})")
                continue
            ep, command = got
            problems = validate(ep)
            if problems:
                print(f"[{i + 1:3d}/{args.episodes}] INVALID: {problems}")
                continue
            path = write_episode(ep, args.out)
            # 사이드카: 내린 명령. 계약 npz 를 안 건드리므로 검증기와 무관하다.
            np.save(path.with_suffix(".command.npy"), command)
            written.append(path)
            n_success += int(ep.meta.success)
            print(
                f"[{i + 1:3d}/{args.episodes}] {'O' if ep.meta.success else 'X'} "
                f"{ep.meta.n_steps:3d} steps  {path.stat().st_size / 1e6:5.2f}MB  {path.name}"
            )

    index = write_dataset_index(
        args.out,
        extra={
            "collected_by": args.author,
            "skill_id": args.skill_id,
            "jitter_m": args.jitter,
            "seed": args.seed,
            "scripted": True,
        },
    )
    print(f"\n에피소드 {len(written)}개 저장 (도달불가로 건너뜀 {skipped}개)")
    print(f"파지 성공 {n_success}/{len(written)}")
    print(f"인덱스: {index}")

    # Round-trip check: a contract that has not been read back is not proven.
    # 왕복 검증: 다시 읽어보지 않은 계약은 증명된 것이 아니다.
    if written:
        back = read_episode(written[0])
        problems = validate(back)
        total_mb = sum(p.stat().st_size for p in written) / 1e6
        print(f"\n왕복 재검증 {written[0].name}: {'통과' if not problems else problems}")
        print(f"  n_steps={back.meta.n_steps} cameras={back.meta.cameras}")
        print(f"  state {back.state.shape} {back.state.dtype}  action {back.action.shape}")
        print(f"  image {back.images[back.meta.cameras[0]].shape} {back.images[back.meta.cameras[0]].dtype}")
        print(f"  총 용량 {total_mb:.1f}MB / {len(written)}ep = {total_mb / len(written):.2f}MB per ep")

        if args.log:
            rec = log_run(
                experiment="collect_sim",
                author=args.author,
                issue="S15P21A103-64",
                conditions={
                    "skill_id": args.skill_id,
                    "episodes_requested": args.episodes,
                    "jitter_m": args.jitter,
                    "seed": args.seed,
                    "contract_version": CONTRACT_VERSION,
                    "config_sha": file_digest(DEFAULT_CONFIG),
                },
                result={
                    "episodes_written": len(written),
                    "skipped_unreachable": skipped,
                    "grasp_success": n_success,
                    "mb_per_episode": round(total_mb / len(written), 3),
                    "roundtrip_valid": not problems,
                },
            )
            print(f"EXP_LOG.jsonl 기록 (git {rec['git_rev']})")


if __name__ == "__main__":
    main()
