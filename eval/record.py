"""Record rollouts to video on a machine with no display.
화면이 없는 기계에서 롤아웃을 영상으로 기록한다.

The measurement tools tell you a policy scored 0%. They do not tell you whether
it stood still, drifted past the object, closed the gripper too early, or knocked
the cube off the table. Those are different failures with different fixes, and
the fastest way to tell them apart is to watch one.
계측 도구는 정책이 0% 를 냈다고 알려준다. 그 정책이 가만히 있었는지, 물체를 지나쳐
갔는지, 그리퍼를 너무 일찍 닫았는지, 큐브를 테이블 밖으로 쳐냈는지는 알려주지
않는다. 전부 다른 실패이고 대책도 다르며, 구분하는 가장 빠른 방법은 하나를 보는
것이다.

⚠️ 이건 보는 도구다. 성공률은 `tools/eval_rollout.py` 에서 나온다. 여기서 고른
   에피소드 몇 편은 표본이 아니다 — 특히 `--only-failures` 는 실패만 골라 담는다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from policy.base import Policy
from policy.baselines import HoldPolicy, ScriptedPickPolicy, ZeroPolicy
from sim.mujoco.build_scene import load_config
from sim.mujoco.env import MujocoPickEnv

POLICY_NAMES = ("scripted", "hold", "zero", "bc")


@dataclass
class Clip:
    """One recorded episode and how it ended.
    기록된 에피소드 하나와 그 결말."""

    seed: int
    success: bool
    ticks: int
    lift_cm: float
    object_xy: tuple[float, float]
    frames: list[np.ndarray]

    def stem(self, policy_name: str) -> str:
        tag = "ok" if self.success else "fail"
        return (
            f"{policy_name}_{tag}_seed{self.seed}_"
            f"xy{self.object_xy[0]:+.3f}{self.object_xy[1]:+.3f}_"
            f"lift{self.lift_cm:04.1f}cm"
        )


def build_policy(name: str, env: MujocoPickEnv, ckpt: Path | None) -> Policy:
    """The same policies the rollout harness scores, so what you watch is what
    was measured.
    롤아웃 harness 가 채점하는 것과 같은 정책. 보는 것과 잰 것이 같아야 한다."""
    if name == "scripted":
        return ScriptedPickPolicy(env)
    if name == "hold":
        return HoldPolicy()
    if name == "zero":
        return ZeroPolicy()
    if name == "bc":
        from policy.act import load_policy

        if ckpt is None:
            raise SystemExit("--policy bc 는 --policy-ckpt 가 필요하다")
        pol = load_policy(ckpt)
        print(f"학습 정책 로드: {pol.describe()}")
        if pol.meta.get("trained_on") == "random_tensors":
            raise SystemExit("랜덤 텐서로 학습된 체크포인트다. 기록할 의미가 없다.")
        return pol
    raise ValueError(f"모르는 정책: {name}")


def record_episode(
    env: MujocoPickEnv,
    policy: Policy,
    seed: int,
    renderer: mujoco.Renderer,
    camera: str,
) -> Clip:
    """Run one episode, keeping every frame.
    에피소드 하나를 돌리며 모든 프레임을 남긴다."""
    obs = env.reset(seed=seed)
    policy.reset(seed=seed)
    obj = env.object_position()
    frames: list[np.ndarray] = []
    success = False
    ticks = 0

    def grab() -> None:
        renderer.update_scene(env.data, camera=camera)
        frames.append(renderer.render().copy())

    grab()
    for _ in range(env.max_ticks):
        obs = env.step(policy.act(obs))
        ticks += 1
        grab()
        if env.is_success():
            success = True
            break

    return Clip(
        seed=seed,
        success=success,
        ticks=ticks,
        lift_cm=env.lift_height() * 100.0,
        object_xy=(round(float(obj[0]), 5), round(float(obj[1]), 5)),
        frames=frames,
    )


def save(clip: Clip, out_dir: Path, policy_name: str, fps: int) -> Path:
    """Write one clip, falling back through formats rather than failing.
    클립 하나를 쓴다. 실패하는 대신 형식을 낮춰 간다.

    A missing video encoder should not cost you the recording — the frames were
    already computed, and a PNG sequence keeps them.
    인코더가 없다고 기록을 잃으면 안 된다. 프레임은 이미 계산됐고 PNG 시퀀스로
    남겨두면 나중에 합칠 수 있다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = clip.stem(policy_name)

    try:
        import imageio.v3 as iio
    except ImportError:
        iio = None  # type: ignore[assignment]

    if iio is not None:
        for ext, kwargs in ((".mp4", {"fps": fps}), (".gif", {"fps": fps, "loop": 0})):
            path = out_dir / f"{stem}{ext}"
            try:
                iio.imwrite(path, np.stack(clip.frames), **kwargs)
                return path
            except Exception:
                continue

    frame_dir = out_dir / stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio2

        for i, f in enumerate(clip.frames):
            iio2.imwrite(frame_dir / f"{i:04d}.png", f)
    except ImportError:
        np.savez_compressed(out_dir / f"{stem}.npz", frames=np.stack(clip.frames))
        return out_dir / f"{stem}.npz"
    return frame_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=POLICY_NAMES, default="bc")
    parser.add_argument("--policy-ckpt", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument(
        "--out", type=Path, default=Path("out/download"),
        help="기본값 out/download — 내려받을 것만 여기 모은다. 그 외 산출물과 섞지 않는다",
    )
    parser.add_argument("--camera", type=str, default="cam_front")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--only-failures",
        action="store_true",
        help="실패한 에피소드만 저장한다. 배울 것이 거기 있다",
    )
    args = parser.parse_args()

    cfg = load_config()
    seeds = [args.seed_base + i for i in range(args.episodes)]
    needs_pixels = args.policy == "bc"

    with MujocoPickEnv(cfg, render=needs_pixels, object_jitter_m=args.jitter) as env:
        policy = build_policy(args.policy, env, args.policy_ckpt)
        cams = env.camera_names
        if args.camera not in cams:
            raise SystemExit(f"카메라 {args.camera!r} 가 없다. 있는 것: {cams}")

        renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
        try:
            print(
                f"정책 {policy.name} · {args.episodes} 에피소드 · 물체 xy "
                f"±{args.jitter * 1000:.0f}mm · 시드 {seeds[0]}~{seeds[-1]} · "
                f"카메라 {args.camera} {args.width}x{args.height}"
            )
            written: list[Path] = []
            n_ok = 0
            for seed in seeds:
                clip = record_episode(env, policy, seed, renderer, args.camera)
                n_ok += int(clip.success)
                mark = "성공" if clip.success else "실패"
                if args.only_failures and clip.success:
                    print(f"  seed {seed}  {mark}  (건너뜀)")
                    continue
                path = save(clip, args.out, policy.name, args.fps)
                written.append(path)
                print(
                    f"  seed {seed}  {mark}  {clip.ticks:3d}틱  "
                    f"상승 {clip.lift_cm:5.2f}cm  →  {path.name}"
                )
        finally:
            renderer.close()

    # A manifest, because a folder of mp4 files is not self-explanatory a week
    # later. The filename carries the outcome; this carries what to look for.
    # 목록 파일을 함께 쓴다. mp4 가 담긴 폴더는 일주일 뒤에 스스로를 설명하지 못한다.
    # 파일명은 결과를 담고, 이 파일은 무엇을 봐야 하는지를 담는다.
    if written:
        manifest = args.out / "목록.txt"
        lines = [
            f"롤아웃 영상 — {policy.name} · {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"조건: 물체 위치 무작위 ±{args.jitter * 1000:.0f}mm · 시드 {seeds[0]}~{seeds[-1]}",
            f"      카메라 {args.camera} · {args.width}x{args.height} · {args.fps}fps",
            "",
            "파일명 읽는 법:",
            "  <정책>_<ok|fail>_seed<시드>_xy<물체위치>_lift<올라간높이>.mp4",
            "  같은 시드끼리는 물체 위치가 같다 — 나란히 틀어 비교할 수 있다",
            "",
            "파일:",
        ]
        for path in written:
            lines.append(f"  {path.name}")
        lines += [
            "",
            "⚠️ 이 영상들은 표본이 아니다. 여기 몇 편을 보고 성공률을 말하면 안 된다.",
            "   성공률은 tools/eval_rollout.py 에서만 나온다 (고정 시드·동일 조건·게이트 판정).",
        ]
        if args.only_failures:
            lines.append("⚠️ 이 실행은 --only-failures 로 실패만 골라 담았다.")
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"목록: {manifest}")

    print(f"\n{n_ok}/{len(seeds)} 성공 · 파일 {len(written)}개 → {args.out}")
    print(
        "⚠️ 여기 담긴 것은 표본이 아니다. 보고할 성공률은 tools/eval_rollout.py 로 낸다"
        + (" — 이 실행은 실패만 골라 담았다." if args.only_failures else ".")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
