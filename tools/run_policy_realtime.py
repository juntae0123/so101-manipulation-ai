"""Closed-loop policy runner: camera -> policy -> IK -> SO-101 joints.
폐루프 러너. 카메라 → 정책 → IK → SO-101 관절.

제품 기조 (이걸 틀리면 나머지가 다 틀어진다)
--------------------------------------------
**결과 모방이지 궤적 모방이 아니다.** 시연 궤적을 그대로 그리는 게 목표가 아니라
같은 물체를 같게 집는 게 목표다. 접근 경로는 IK 가 스스로 푼다.

그래서 이 도구는 **재관측한다.** 8스텝 중 [1,5) 만 보내고 다시 본다(0.4초마다).
전체 궤적을 미리 풀어 한 번에 재생하면 오차가 누적되고, 무엇보다 **궤적 모방을
재는 셈이 된다.**

⚠️ 판정은 궤적 오차가 아니라 **파지 성공**이다 — 물체가 10cm 이상 올라가고 0.5초 유지
(D-AI-58 의 1b 게이트).

카메라 소스 — 하나만 고른다
---------------------------
    --camera v4l2:0                 USB 캠 / 폰을 웹캠으로 붙인 경우
    --camera dir:/path/frames       폴더의 프레임을 순서대로 (폰이 올려주는 방식)
    --camera zarr:/path.zarr.zip:3  기록된 시연 프레임 (**로봇 없이 배선 전체 검증용**)

`zarr:` 로 `--no-robot --dry-run` 을 돌리면 카메라도 팔도 없이 전 구간이 도는지 본다.
월요일에 카메라만 꽂으면 되게 하려면 이걸 먼저 통과시킨다.

전처리 — evaluate.py 와 같은 경로를 쓴다
----------------------------------------
    tf = get_image_transform(in_res=(W, H), out_res=(224, 224))
    rgb = draw_predefined_mask(rgb, color=(0,0,0), mirror=False, gripper=True, finger=False)
    obs = get_real_umi_obs_dict(raw, shape_meta, obs_pose_repr=cfg..., episode_start_pose=[start])
    absolute = get_real_umi_action(action, raw, action_pose_repr=cfg...)   # ← 'relative'

⚠️ `action_pose_repr` 를 안 넘기면 기본값 `'abs'` 로 떨어져 팔이 원점으로 간다.
   `'rel'` 은 소스가 스스로 legacy buggy 라 적은 별개 경로다. **셋 다 에러 없이 돈다.**

Usage
-----
  python run_policy_realtime.py --selftest
  # 배선 검증 (로봇·카메라 없이)
  python run_policy_realtime.py --checkpoint CK --task configs/can_side.yaml \
      --camera zarr:outputs/ds_f0918_42_holdout.zarr.zip:0 --no-robot --dry-run --steps 12
  # 실물
  python run_policy_realtime.py --checkpoint CK --task configs/can_side.yaml \
      --camera v4l2:0 --port /dev/ttyTHS1 --yes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ACTION_STEPS = 4                 # evaluate.py 기본값. index [1, 1+4)
MAX_STEP_M = 0.035               # evaluate.py 의 컨트롤러 상한. 한 번에 이만큼만 움직인다
Z_CLIP = (0.006, 0.40)


# ──────────────────────────────────────────────── 카메라 소스

class DirSource:
    """Frames from a directory, in sorted order. 폴더의 프레임을 정렬 순서로."""

    def __init__(self, path: str) -> None:
        self.files = sorted(Path(path).glob("*.[jp][pn]g"))
        if not self.files:
            raise SystemExit(f"!! {path} 에 jpg/png 가 없다 (찾음 0 / 전체 "
                             f"{len(list(Path(path).glob('*')))})")
        self.i = 0
        print(f"[카메라] dir {path} · 프레임 {len(self.files)}장")

    def read(self):
        import cv2
        if self.i >= len(self.files):
            return None
        img = cv2.imread(str(self.files[self.i]))
        self.i += 1
        return None if img is None else img[:, :, ::-1].copy()


class ZarrSource:
    """Recorded demo frames. 기록된 시연 프레임. 로봇 없이 배선을 검증한다."""

    def __init__(self, spec: str) -> None:
        path, _, ep = spec.rpartition(":")
        import zarr
        z = zarr.open(path, mode="r")
        ends = np.asarray(z["meta"]["episode_ends"][:])
        e = int(ep)
        if not (0 <= e < len(ends)):
            raise SystemExit(f"!! 에피소드 {e} 가 범위 밖 (편 {len(ends)}개)")
        self.lo = 0 if e == 0 else int(ends[e - 1])
        self.hi = int(ends[e])
        self.arr = z["data"]["camera0_rgb"]
        self.pos = z["data"]["robot0_eef_pos"]
        self.rot = z["data"]["robot0_eef_rot_axis_angle"]
        self.gw = z["data"]["robot0_gripper_width"]
        self.i = self.lo
        self.last = self.lo
        print(f"[카메라] zarr {path} 편 {e}/{len(ends)} · 행 {self.lo}~{self.hi} "
              f"({self.hi - self.lo}프레임)")

    def state(self, row: int) -> dict:
        """Recorded proprioception for one row. 그 행의 기록된 자기수용 상태."""
        return {"robot0_eef_pos": np.asarray(self.pos[row], float),
                "robot0_eef_rot_axis_angle": np.asarray(self.rot[row], float),
                "robot0_gripper_width": np.asarray(self.gw[row], float).reshape(1)}

    def read(self):
        if self.i >= self.hi:
            return None
        self.last = self.i
        img = np.asarray(self.arr[self.i])
        self.i += 1
        if img.dtype != np.uint8:                      # 정규화돼 있으면 되돌린다
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[0] in (1, 3) and img.ndim == 3:   # CHW 로 저장된 경우
            img = np.transpose(img, (1, 2, 0))
        return img


class V4l2Source:
    def __init__(self, idx: int) -> None:
        import cv2
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            raise SystemExit(f"!! 카메라 {idx} 를 못 연다")
        print(f"[카메라] v4l2 {idx}")

    def read(self):
        ok, bgr = self.cap.read()
        return bgr[:, :, ::-1].copy() if ok else None


def parse_camera(spec: str):
    """One of v4l2:<i> / dir:<path> / zarr:<path>:<ep>. 셋 중 하나만."""
    kind, _, rest = spec.partition(":")
    if not rest:
        raise SystemExit(f"!! --camera 형식이 아니다: {spec!r}. "
                         "v4l2:0 / dir:/path / zarr:/path.zarr.zip:0")
    if kind == "v4l2":
        return V4l2Source(int(rest))
    if kind == "dir":
        return DirSource(rest)
    if kind == "zarr":
        return ZarrSource(rest)
    raise SystemExit(f"!! 모르는 카메라 종류 {kind!r}. v4l2 / dir / zarr 중 하나")


# ──────────────────────────────────────────────── 관측 조립

def stack_history(hist: list[dict]) -> dict:
    """Stack a deque of per-step obs into (T, ...) arrays. 스텝별 관측을 (T, ...) 로 쌓는다."""
    if not hist:
        raise ValueError("관측 이력이 비었다")
    keys = set(hist[-1])
    for h in hist:
        if set(h) != keys:
            raise ValueError(f"이력마다 키가 다르다: {sorted(keys)} vs {sorted(h)}")
    return {k: np.stack([h[k] for h in hist]) for k in keys}


def clamp_target(target_pos: np.ndarray, current_pos: np.ndarray) -> np.ndarray:
    """Bound one commanded step, exactly as evaluate.py does. evaluate.py 와 같은 상한."""
    delta = np.asarray(target_pos, float) - np.asarray(current_pos, float)
    n = float(np.linalg.norm(delta))
    if n > MAX_STEP_M:
        delta = delta * (MAX_STEP_M / n)
    out = np.asarray(current_pos, float) + delta
    out[2] = float(np.clip(out[2], *Z_CLIP))
    return out


# ──────────────────────────────────────────────── 자체검증

def selftest() -> int:
    ok = total = 0

    def check(name, cond, detail=""):
        nonlocal ok, total
        total += 1
        ok += bool(cond)
        print(f"[{total}] {name:<46} {'OK' if cond else '!! 실패'}  {detail}")

    for bad in ("v4l2", "", "webcam:0", "zarr:onlypath"):
        try:
            parse_camera(bad)
            died = False
        except SystemExit:
            died = True
        except Exception:                              # noqa: BLE001
            died = True
        if not died:
            check(f"잘못된 --camera {bad!r} 거부", False)
            break
    else:
        check("잘못된 --camera 4종 전부 거부", True)

    h = [{"a": np.zeros(3), "b": np.ones(1)} for _ in range(2)]
    s = stack_history(h)
    check("이력 2개 → (2, ...) 로 쌓임", s["a"].shape == (2, 3) and s["b"].shape == (2, 1),
          str({k: v.shape for k, v in s.items()}))

    try:
        stack_history([{"a": np.zeros(3)}, {"b": np.zeros(3)}])
        raised = False
    except ValueError:
        raised = True
    check("이력 키가 다르면 거부", raised)

    try:
        stack_history([])
        raised = False
    except ValueError:
        raised = True
    check("빈 이력 거부", raised)

    cur = np.array([0.40, 0.0, 0.05])
    far = cur + np.array([0.2, 0.0, 0.0])
    out = clamp_target(far, cur)
    check("한 스텝 상한 35mm", abs(np.linalg.norm(out - cur) - MAX_STEP_M) < 1e-12,
          f"{np.linalg.norm(out - cur) * 1000:.1f} mm")

    near = cur + np.array([0.01, 0.0, 0.0])
    out = clamp_target(near, cur)
    check("상한 안이면 그대로 (판별력)", np.allclose(out, near),
          f"{np.linalg.norm(out - cur) * 1000:.1f} mm")

    # ⚠️ 순서가 있다 — 35mm 상한이 **먼저**, z 클립이 나중이다 (evaluate.py 와 동일).
    #    초판은 z=-0.5 로 검사했는데 상한이 먼저 걸려 0.015 에서 멈춘다. 클립이
    #    물리지 않는 입력으로 클립을 검사한 셈이었다. 상한 안쪽 입력으로 바꾼다.
    low = clamp_target(np.array([0.40, 0.0, 0.0]), np.array([0.40, 0.0, 0.02]))
    check("z 하한으로 클립 (상한 안쪽 입력)", abs(low[2] - Z_CLIP[0]) < 1e-12,
          f"{low[2]:.4f} m  기대 {Z_CLIP[0]}")

    deep = clamp_target(np.array([0.40, 0.0, -0.5]), cur)
    check("상한이 z 클립보다 먼저 (판별력)", abs(deep[2] - (cur[2] - MAX_STEP_M)) < 1e-12,
          f"{deep[2]:.4f} m — 35mm 만 내려간다")

    # 액션 규약은 policy_to_joints 와 **같은 구현을 쓴다**. 두 벌이면 갈린다
    try:
        import policy_to_joints as p2j
        ident = np.zeros((8, 10))
        ident[:, 3], ident[:, 7], ident[:, 9] = 1.0, 1.0, 0.05
        T = np.eye(4); T[:3, 3] = [0.4, 0, 0.05]
        poses, gaps, diag = p2j.decode_chunk(ident, T)
        check("policy_to_joints 규약 공유 (항등 → 불변)",
              max(np.abs(q - T).max() for q in poses) < 1e-12 and diag["emitted"] == 4)
        check("실행 구간이 [1,5) 로 같다", diag["exec_slice"] == [1, 5])
    except ImportError as e:
        check(f"policy_to_joints import ({e})", False)

    try:
        from smoke_deploy_ckpt import _prepare_umi_path            # noqa: F401
        check("UMI 경로 해결기를 재사용한다 (두 벌 금지)", True)
    except ImportError as e:
        check(f"smoke_deploy_ckpt._prepare_umi_path import ({e})", False)

    check("프레임 stride 기본값 = 실행 스텝 수 (시간축 일치)", ACTION_STEPS == 4,
          f"{ACTION_STEPS}")

    print(f"\n자체검증 {ok} / {total}")
    return 0 if ok == total else 1


# ──────────────────────────────────────────────── 본체

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--task")
    ap.add_argument("--camera", help="v4l2:0 / dir:/path / zarr:/path.zarr.zip:0")
    ap.add_argument("--port", default="/dev/ttyTHS1")
    ap.add_argument("--no-robot", action="store_true",
                    help="팔 없이 돈다. 관절 상태를 IK 결과로 이어붙여 배선만 검증")
    ap.add_argument("--dry-run", action="store_true", help="계산만. 서보에 아무것도 안 보낸다")
    ap.add_argument("--steps", type=int, default=40, help="재관측 횟수 상한")
    ap.add_argument("--action-steps", type=int, default=ACTION_STEPS)
    ap.add_argument("--jaw-offset-deg", type=float, default=0.0)
    ap.add_argument("--frame-stride", type=int, default=None,
                    help="재관측 1회당 넘길 프레임 수. 기본 = --action-steps. "
                         "기록 소스(zarr/dir)의 시간축을 실행 주기와 맞춘다")
    ap.add_argument("--ik-reject-mm", type=float, default=None,
                    help="IK 위치 잔차가 이보다 크면 그 스텝을 버린다. "
                         "기본 없음 — evaluate.py 는 거부하지 않는다(97%를 낸 그 경로). "
                         "실물에서 보수적으로 가고 싶을 때만 준다")
    ap.add_argument("--out", default=None)
    ap.add_argument("--umi-root", help="공식 UMI 저장소 경로 (diffusion_policy 의 부모)")
    ap.add_argument("--obs-from-zarr", action="store_true",
                    help="관측(자기수용)도 zarr 기록에서 준다. 교사강제. "
                         "팔 상태와 화면이 갈리지 않아 **정책이 닫는지**를 순수하게 본다. "
                         "제어를 재는 게 아니라 정책 출력을 재는 모드다")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    print("자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 팔을 건드리지 않는다")
    print()
    for need in ("checkpoint", "task", "camera"):
        if not getattr(a, need):
            ap.error(f"--{need} 가 필요하다")

    # ⚠️ 공식 UMI(`diffusion_policy`)를 sys.path 에 먼저 올린다. 안 하면 hydra 가
    #    'Error locating target' 로 죽는데 그건 체크포인트 문제가 아니라 환경 문제다.
    #    smoke_deploy_ckpt 의 해결기를 **재사용한다** — 두 벌이면 갈린다.
    from smoke_deploy_ckpt import _prepare_umi_path
    print(f"[{_prepare_umi_path(a.umi_root)}]")

    import torch, dill, hydra                                        # noqa: E401
    import policy_to_joints as p2j

    # 1) 정책
    payload = torch.load(a.checkpoint, map_location="cpu", pickle_module=dill,
                         weights_only=False)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    sd = payload["state_dicts"]
    key = "ema_model" if "ema_model" in sd else "model"
    policy.load_state_dict(sd[key])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(dev).eval()
    print(f"[정책] {key} · {dev} · horizon {cfg.shape_meta.action.horizon} · "
          f"실행 [1,{1 + a.action_steps})")

    obs_repr = cfg.task.pose_repr.obs_pose_repr
    act_repr = cfg.task.pose_repr.action_pose_repr
    print(f"[규약] obs_pose_repr={obs_repr} · action_pose_repr={act_repr}")
    if act_repr != "relative":
        raise SystemExit(f"!! action_pose_repr 가 {act_repr!r} 다. 'relative' 가 아니면 "
                         "상대를 절대로 해석한다. 중단")

    # 2) 로봇 모델 (IK·FK) + 파지점 게이트
    from simulation.env import PickEnv
    env = p2j.make_env(PickEnv, a.task)
    p2j.tcp_gate(p2j.probe_tcp_offset(env), None)

    from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action
    from umi.common.cv_util import draw_predefined_mask, get_image_transform

    cam = parse_camera(a.camera)
    first = cam.read()
    if first is None:
        raise SystemExit("!! 카메라에서 첫 프레임을 못 받았다")
    h, w = first.shape[:2]
    tf = get_image_transform(in_res=(w, h), out_res=(224, 224))
    print(f"[카메라] 입력 {w}x{h} → 224x224")

    # 3) 로봇
    bus = None
    if not a.no_robot:
        import replay_trajectory as rt
        bus = rt.Bus(a.port)
        if bus.position(200) is not None:
            bus.close(); raise SystemExit("!! 없는 ID 200 이 응답했다. 버스 이상. 중단")
        q_meas, g_meas, okn = rt.read_all(bus)
        if okn < len(rt.JOINTS):
            bus.close(); raise SystemExit(f"!! 관절 {okn}/{len(rt.JOINTS)} 만 읽힌다. 중단")
        q_cur = np.asarray(q_meas, float)
        gap_cur = 0.09 if g_meas is None else float(g_meas)
        print(f"[로봇] 현재 자세 읽음 · gap {gap_cur * 1000:.1f} mm")
        if not (a.dry_run or a.yes):
            if input("팔 주변을 비웠나? 진행하려면 'go': ").strip() != "go":
                bus.close(); raise SystemExit("중단")
    else:
        q_cur = np.asarray(env.home_q, float)
        gap_cur = 0.09
        print("[로봇] --no-robot · 홈 자세에서 시작하고 IK 결과를 이어붙인다")

    import mujoco
    def fk(q):
        env.ik_data.qpos[env.qids] = q
        mujoco.mj_forward(env.model, env.ik_data)
        return env.tcp(env.ik_data).copy()

    from scipy.spatial.transform import Rotation
    T0 = fk(q_cur)
    episode_start = np.r_[T0[:3, 3], Rotation.from_matrix(T0[:3, :3]).as_rotvec()]

    teacher = a.obs_from_zarr
    if teacher and not isinstance(cam, ZarrSource):
        raise SystemExit("!! --obs-from-zarr 는 --camera zarr: 일 때만 된다")
    if teacher:
        print("[모드] 교사강제 — 자기수용도 기록에서 준다. 제어가 아니라 "
              "**정책이 닫는가**를 잰다")

    def observe(rgb):
        masked = draw_predefined_mask(rgb.copy(), color=(0, 0, 0), mirror=False,
                                      gripper=True, finger=False)
        if teacher:
            st = cam.state(cam.last)
            return {"camera0_rgb": tf(masked), **st}
        T = fk(q_cur)
        return {"camera0_rgb": tf(masked),
                "robot0_eef_pos": T[:3, 3].copy(),
                "robot0_eef_rot_axis_angle": Rotation.from_matrix(T[:3, :3]).as_rotvec(),
                "robot0_gripper_width": np.array([gap_cur])}

    hist = [observe(first)]
    hist.append(hist[0])
    log = []
    sent = 0
    rejected = 0
    gap_track = []
    try:
        for it in range(a.steps):
            raw = stack_history(hist[-2:])
            conv = get_real_umi_obs_dict(raw, cfg.shape_meta, obs_pose_repr=obs_repr,
                                         episode_start_pose=[episode_start])
            missing = set(cfg.shape_meta.obs) - set(conv)
            if missing:
                raise SystemExit(f"!! 관측 키 누락 {sorted(missing)} "
                                 f"({len(conv)} / {len(cfg.shape_meta.obs)})")
            batch = {k: torch.from_numpy(v.astype(np.float32))[None].to(dev)
                     for k, v in conv.items()}
            with torch.inference_mode():
                action = policy.predict_action(batch)["action"][0].cpu().numpy()
            absolute = get_real_umi_action(action, raw, action_pose_repr=act_repr)
            if teacher:
                gt_gap = float(cam.state(cam.last)["robot0_gripper_width"][0])
                pred_gap = float(np.clip(absolute[1][6], 0.0, 0.09))
                gap_track.append((cam.last, gt_gap * 1000, pred_gap * 1000))

            for target in absolute[1:1 + a.action_steps]:
                T_now = fk(q_cur)
                pos = clamp_target(target[:3], T_now[:3, 3])
                Rm = p2j.apply_jaw_offset(
                    Rotation.from_rotvec(target[3:6]).as_matrix(), a.jaw_offset_deg)
                try:
                    q_new = env.ik(pos, Rm, seed=q_cur, strict=False)
                except Exception as exc:                # noqa: BLE001
                    print(f"  [{it}] IK 예외 {type(exc).__name__} — 이 스텝 건너뜀")
                    continue
                if q_new is None:
                    print(f"  [{it}] IK 해 없음 — 건너뜀")
                    continue
                Tc = fk(q_new)
                res = float(np.linalg.norm(Tc[:3, 3] - pos))
                if a.ik_reject_mm is not None and res * 1000 > a.ik_reject_mm:
                    print(f"  [{it}] IK 잔차 {res * 1000:.1f}mm > {a.ik_reject_mm:.0f}mm — 건너뜀")
                    rejected += 1
                    continue
                gap_new = float(np.clip(target[6], 0.0, 0.09))

                if bus is not None and not a.dry_run:
                    import replay_trajectory as rt
                    rt.preflight([list(q_new)], [gap_new])
                    for j, r in zip(rt.JOINTS, q_new):
                        bus.move(j[1], rt.joint_to_tick(j, r))
                    bus.move(rt.GRIPPER[1], rt.gripper_to_tick(gap_new))
                    sent += 1
                q_cur, gap_cur = np.asarray(q_new, float), gap_new
                log.append({"iter": it, "pos": pos.tolist(), "gap_m": gap_new,
                            "q": [float(v) for v in q_new], "ik_residual_m": res})

            stride = a.frame_stride if a.frame_stride is not None else a.action_steps
            nxt = None
            for _ in range(stride):                 # 실행한 시간만큼 프레임을 넘긴다
                f = cam.read()
                if f is None:
                    break
                nxt = f
            if nxt is None:
                print(f"  프레임 끝 (반복 {it + 1}/{a.steps})")
                break
            hist.append(observe(nxt))
            hist = hist[-2:]
            if bus is not None and not a.dry_run:
                time.sleep(0.4)          # 재관측 주기 (실행 4스텝 × 0.1초)
    except KeyboardInterrupt:
        print("\n!! 중단됨")
    finally:
        if bus is not None:
            bus.close()

    print(f"\n반복 {len(set(r['iter'] for r in log))} · 명령 스텝 {len(log)} · "
          f"서보 전송 {sent}")
    if log:
        r = [x["ik_residual_m"] * 1000 for x in log]
        g = [x["gap_m"] * 1000 for x in log]
        print(f"IK 잔차  중앙 {np.median(r):.2f}mm  최대 {max(r):.2f}mm  "
              f"(거부 {rejected} / 시도 {len(log) + rejected})")
        print(f"gap      {min(g):.1f} ~ {max(g):.1f} mm")
        obj = 41.0
        if min(g) <= obj + 10:
            print(f"         → 물체 폭 {obj:.0f}mm 근처까지 닫힌다. 파지 동작이 있다")
        else:
            print(f"         → 최소 {min(g):.1f}mm 로 물체 폭 {obj:.0f}mm 보다 "
                  f"{min(g)-obj:.1f}mm 넓다. **이 구간에는 파지가 없다** "
                  f"(에피소드 끝까지 안 갔거나, 정책이 안 닫는 것이다)")
    else:
        print("!! 명령이 하나도 안 나갔다. IK 가 전부 거부됐거나 프레임이 없다")
    if gap_track:
        rows = [r for r, _, _ in gap_track]
        gt = np.array([g for _, g, _ in gap_track])
        pr = np.array([p for _, _, p in gap_track])
        print(f"\n=== 교사강제 gap 궤적 · {len(gap_track)} 관측 (행 {rows[0]}~{rows[-1]}) ===")
        print(f"  기록 {gt.min():.1f} ~ {gt.max():.1f} mm   (닫힘폭 {gt.max()-gt.min():.1f})")
        print(f"  예측 {pr.min():.1f} ~ {pr.max():.1f} mm   (닫힘폭 {pr.max()-pr.min():.1f})")
        print(f"  절대오차 중앙 {np.median(np.abs(pr-gt)):.2f} mm  최대 {np.abs(pr-gt).max():.2f} mm")
        obj = 41.0
        closes = pr.min() <= obj + 10
        print(f"  판정: 정책이 물체 폭 {obj:.0f}mm 근처까지 닫는가 → "
              f"{'닫는다' if closes else '**안 닫는다**'}")
        if gt.max() - gt.min() < 5:
            print("  ⚠️ 기록 자체의 닫힘폭이 5mm 미만이다. 이 편으로는 판정하지 마라")
        print("  " + "  ".join(f"{r}:{g:.0f}/{p:.0f}" for r, g, p in gap_track[:12]))
        print("  (행:기록/예측 mm)")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"log": log, "gap_track": gap_track}, indent=1), encoding="utf-8")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
