"""Does the policy stop on its own when the object is not there?
대상물이 없을 때 정책이 스스로 멈추는가.

사전등록: AI/docs/PREREG_empty_scene_refusal_0919.md (실행 전 작성)
이슈: S15P21A103-113, -36

무엇을 재나 / What is measured
------------------------------
같은 책상을 물체 있는 상태 / 없는 상태로 찍은 프레임을 정책에 넣고,
예측 8스텝 중 **gap 최소값**을 비교한다. 물체가 있을 때만 닫으면 정책 자신이
거부 신호를 주는 것이고, 둘 다 닫으면 정책은 장면을 안 보는 것이다.

⚠️ **판정 대상은 파지 순간의 개구다. 궤적이 아니다.**

이 계측기가 스스로를 못 믿는 지점 / Where this instrument refuses itself
----------------------------------------------------------------------
1. **빈 장면 검정** — with/without 짝 프레임의 평균 절대 화소차가 임계 미만이면
   "빈 장면"이 사실 같은 그림이다. 그러면 차이 0mm 가 **공짜로** 나온다.
   그 경우 결과를 내지 않고 죽는다.
2. **정답 아는 행** — with 조건에서 정책이 안 닫으면 배선이 틀린 것이다.
   그 경우에도 결과를 내지 않는다. "없음"과 "괜찮음"을 가른다.
3. **모수를 같이 찍는다** — 읽은 프레임 수 / 쓴 프레임 수 / 짝지은 수.

Usage
-----
  # [서버]
  ~/envs/handoff312/bin/python AI/tools/probe_empty_scene_refusal.py --selftest
  ~/envs/handoff312/bin/python AI/tools/probe_empty_scene_refusal.py --ckpt ~/handoff/deploy/so101_pick_v1.ckpt --with-dir ~/frames_with --without-dir ~/frames_without --out ~/handoff/outputs/empty_scene.json

프레임 만드는 법 / How to capture the frames
-------------------------------------------
폰을 삼각대에 고정하고 같은 자세에서 두 번 찍는다. 책상 위 물체만 빼고 나머지는
건드리지 않는다. 카메라를 움직이면 이 측정은 무효다 (빈 장면 검정이 잡아주지만,
잡히면 다시 찍어야 한다).

되돌리기 / Reverting
--------------------
이 파일을 지우면 끝이다. 체크포인트·설정·데이터를 하나도 안 건드린다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PIXEL_DIFF_MIN = 8.0        # 0~255. 사전등록값
WITH_CLOSE_MAX_MM = 50.0    # 정답 아는 행: 물체 있으면 이보다 작게 닫혀야 한다
GATE_PASS_MM = 20.0
GATE_AMBIG_MM = 5.0
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_frames(d: Path) -> tuple[list[np.ndarray], int, int]:
    """Read every image in sorted order. Return frames, found, readable.
    정렬 순서로 전부 읽는다. 프레임 · 찾은 수 · 읽힌 수를 같이 돌려준다."""
    import cv2
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
    frames = []
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is not None:
            frames.append(bgr[:, :, ::-1].copy())
    return frames, len(files), len(frames)


def pixel_gap(a: list[np.ndarray], b: list[np.ndarray]) -> tuple[float, int]:
    """Mean absolute pixel difference over paired frames, and how many pairs.
    짝지은 프레임의 평균 절대 화소차와 짝 수."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0
    diffs = []
    for i in range(n):
        x, y = a[i].astype(np.float64), b[i].astype(np.float64)
        if x.shape != y.shape:
            raise SystemExit(f"!! {i}번째 짝의 해상도가 다르다 {x.shape} vs {y.shape}. "
                             "같은 카메라로 다시 찍어라")
        diffs.append(float(np.abs(x - y).mean()))
    return float(np.mean(diffs)), n


def verdict(delta_mm: float) -> tuple[str, str]:
    """Pre-registered gate. Nothing here is decided after seeing the data.
    사전등록 게이트. 데이터를 보고 정하는 값이 여기 하나도 없다."""
    if delta_mm >= GATE_PASS_MM:
        return "A_PASS", "정책이 스스로 신호를 준다. 러너에 거부 게이트를 넣는다"
    if delta_mm >= GATE_AMBIG_MM:
        return "AMBIGUOUS", "애매하다. n 을 조건당 60으로 올린다. 게이트를 낮추지 않는다"
    return "A_FAIL", "정책은 장면을 안 본다. VLM ③ 로 간다. FE 표시도 병행"


def selftest() -> int:
    """Known-answer rows plus deliberately wrong inputs.
    정답 아는 행 + 고의로 틀린 입력."""
    ok = tot = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok, tot
        tot += 1
        ok += bool(cond)
        print(f"  {'OK ' if cond else '!! '} [{tot}] {label:<44} {detail}")

    rng = np.random.default_rng(0)
    # 0~200 으로 둔다. +40 이 255 에서 잘리면 기대값 40 이 안 나온다 —
    # 초판에서 이걸로 [2] 가 37.19 로 떨어졌고, 계측기가 내 기대값을 잡아냈다.
    base = rng.integers(0, 200, (16, 16, 3), dtype=np.uint8)
    same = [base.copy() for _ in range(4)]
    differ = [np.clip(base.astype(int) + 40, 0, 255).astype(np.uint8) for _ in range(4)]

    d0, n0 = pixel_gap(same, same)
    check("같은 그림끼리는 화소차 0", abs(d0) < 1e-9 and n0 == 4, f"{d0:.3f} · 짝 {n0}")
    d1, _ = pixel_gap(same, differ)
    check("40계조 밀면 화소차 40", abs(d1 - 40.0) < 1e-6, f"{d1:.3f}")
    check("판별력: 같은 그림은 빈 장면 검정을 통과 못 한다", d0 < PIXEL_DIFF_MIN)
    check("판별력: 다른 그림은 통과한다", d1 >= PIXEL_DIFF_MIN)
    check("짝 수는 짧은 쪽을 따른다", pixel_gap(same, differ[:2])[1] == 2)

    check("게이트 25mm -> A_PASS", verdict(25.0)[0] == "A_PASS")
    check("게이트 12mm -> AMBIGUOUS", verdict(12.0)[0] == "AMBIGUOUS")
    check("게이트 2mm -> A_FAIL", verdict(2.0)[0] == "A_FAIL")
    check("게이트 경계 20.0 은 통과쪽", verdict(20.0)[0] == "A_PASS")
    check("게이트 경계 5.0 은 애매쪽", verdict(5.0)[0] == "AMBIGUOUS")

    try:
        pixel_gap([np.zeros((4, 4, 3), np.uint8)], [np.zeros((5, 5, 3), np.uint8)])
        died = False
    except SystemExit:
        died = True
    check("판별력: 해상도가 다르면 죽는다", died)

    print(f"자체검증 {ok} / {tot}")
    return 0 if ok == tot else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", help="배포 체크포인트")
    ap.add_argument("--umi-root", default="~/handoff")
    ap.add_argument("--task", help="시뮬 task yaml — 고정 자기수용값을 여기서 얻는다")
    ap.add_argument("--with-dir", help="대상물 있는 프레임 디렉터리")
    ap.add_argument("--without-dir", help="대상물 없는 프레임 디렉터리")
    ap.add_argument("--out", default="empty_scene_refusal.json")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    for need in ("ckpt", "with_dir", "without_dir"):
        if not getattr(a, need):
            raise SystemExit(f"!! --{need.replace('_','-')} 가 필요하다 (또는 --selftest)")

    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 판정을 내지 않는다")
    print()

    fw, found_w, read_w = load_frames(Path(a.with_dir).expanduser())
    fo, found_o, read_o = load_frames(Path(a.without_dir).expanduser())
    print(f"with    프레임 읽힘 {read_w} / 찾음 {found_w}")
    print(f"without 프레임 읽힘 {read_o} / 찾음 {found_o}")
    if read_w == 0 or read_o == 0:
        raise SystemExit("!! 한쪽이 0장이다. 경로가 맞는지 먼저 본다 "
                         "(0장과 '차이 없음'은 다른 상태다)")

    dpx, pairs = pixel_gap(fw, fo)
    print(f"\n빈 장면 검정: 평균 절대 화소차 {dpx:.2f} / 255 · 짝 {pairs}쌍 "
          f"(임계 {PIXEL_DIFF_MIN})")
    if dpx < PIXEL_DIFF_MIN:
        raise SystemExit(
            "!! 두 조건이 사실상 같은 그림이다. 물체를 실제로 뺐는지, 파일을 섞지\n"
            "   않았는지 확인해라. 이 상태로 재면 '차이 0mm' 가 공짜로 나온다. 중단")

    if not a.task:
        raise SystemExit("!! --task 가 필요하다 (고정 자기수용값을 시뮬 홈에서 가져온다)")

    from smoke_deploy_ckpt import _prepare_umi_path      # 경로 해결기를 재사용한다
    print(f"[{_prepare_umi_path(Path(a.umi_root).expanduser())}]")

    import dill                                                      # noqa: E401
    import hydra
    import torch
    from diffusion_policy.common.pose_repr_util import (  # type: ignore
        get_real_umi_action, get_real_umi_obs_dict)
    from scipy.spatial.transform import Rotation
    from umi.common.cv_util import draw_predefined_mask  # type: ignore

    payload = torch.load(a.ckpt, map_location="cpu", pickle_module=dill,
                         weights_only=False)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    sd = payload["state_dicts"]
    key = "ema_model" if "ema_model" in sd else "model"
    policy.load_state_dict(sd[key])
    dev = a.device if torch.cuda.is_available() else "cpu"
    policy.to(dev).eval()
    obs_repr = cfg.task.pose_repr.obs_pose_repr
    act_repr = cfg.task.pose_repr.action_pose_repr
    print(f"[정책] {key} · {dev} · obs_repr={obs_repr} · action_repr={act_repr}")
    if act_repr != "relative":
        raise SystemExit(f"!! action_pose_repr 가 {act_repr!r} 다. 중단")

    # 고정 자기수용값 — 양 조건에 **같은 값**을 준다. 이미지 기여도만 격리한다.
    from simulation.env import PickEnv
    import mujoco
    env = PickEnv(task_path=str(Path(a.task).expanduser()))
    env.reset(0)
    T0 = env.tcp().copy()
    proprio = {
        "robot0_eef_pos": T0[:3, 3].copy(),
        "robot0_eef_rot_axis_angle": Rotation.from_matrix(T0[:3, :3]).as_rotvec(),
        "robot0_gripper_width": np.array([0.09]),
    }
    episode_start = np.r_[T0[:3, 3], proprio["robot0_eef_rot_axis_angle"]]
    print(f"[고정 자기수용] eef {np.round(T0[:3,3], 4)} · gap 90.0mm (양 조건 동일)")

    shape = tuple(cfg.shape_meta.obs["camera0_rgb"]["shape"])   # (C, H, W)
    import cv2

    def to_obs(rgb: np.ndarray) -> dict:
        masked = draw_predefined_mask(rgb.copy(), color=(0, 0, 0), mirror=False,
                                      gripper=True, finger=False)
        img = cv2.resize(masked, (shape[2], shape[1]))
        chw = np.moveaxis(img, -1, 0).astype(np.float32) / 255.0
        return {"camera0_rgb": chw, **{k: v.copy() for k, v in proprio.items()}}

    def min_gap_mm(frames: list[np.ndarray]) -> list[float]:
        out = []
        for rgb in frames:
            o = to_obs(rgb)
            raw = {k: np.stack([v, v]) for k, v in o.items()}      # 이력 2스텝 동일
            conv = get_real_umi_obs_dict(raw, cfg.shape_meta, obs_pose_repr=obs_repr,
                                         episode_start_pose=[episode_start])
            missing = set(cfg.shape_meta.obs) - set(conv)
            if missing:
                raise SystemExit(f"!! 관측 키 누락 {sorted(missing)} "
                                 f"({len(conv)} / {len(cfg.shape_meta.obs)})")
            batch = {k: torch.from_numpy(v.astype(np.float32))[None].to(dev)
                     for k, v in conv.items()}
            with torch.inference_mode():
                act = policy.predict_action(batch)["action"][0].cpu().numpy()
            absolute = get_real_umi_action(act, raw, action_pose_repr=act_repr)
            out.append(float(np.clip(absolute[:, 6], 0.0, 0.09).min()) * 1000.0)
        return out

    gw = min_gap_mm(fw)
    go = min_gap_mm(fo)
    med_w, med_o = float(np.median(gw)), float(np.median(go))
    print(f"\nwith    gap 최소 중앙 {med_w:.1f} mm   범위 {min(gw):.1f}~{max(gw):.1f}  n={len(gw)}")
    print(f"without gap 최소 중앙 {med_o:.1f} mm   범위 {min(go):.1f}~{max(go):.1f}  n={len(go)}")

    if med_w >= WITH_CLOSE_MAX_MM:
        raise SystemExit(
            f"!! 정답 아는 행 실패 — 물체가 있는데 중앙 {med_w:.1f}mm 로 안 닫는다\n"
            f"   (기대 {WITH_CLOSE_MAX_MM}mm 미만). 배선이나 프레임이 틀렸다. 판정 안 낸다")

    delta = med_o - med_w
    code, nxt = verdict(delta)
    print(f"\n차이(without − with) {delta:.1f} mm   →  {code}")
    print(f"   {nxt}")

    out = {
        "_PREREG": "AI/docs/PREREG_empty_scene_refusal_0919.md (실행 전 작성)",
        "_criterion": "파지 순간 개구. 궤적 아님. 자기수용값은 양 조건 동일",
        "_limits": "같은 장면 연속 촬영이라 독립 표본 아님 · 조명/배경 1종",
        "frames": {"with_found": found_w, "with_read": read_w,
                   "without_found": found_o, "without_read": read_o, "pairs": pairs},
        "pixel_diff_mean": round(dpx, 3),
        "with_min_gap_mm": [round(x, 2) for x in gw],
        "without_min_gap_mm": [round(x, 2) for x in go],
        "median_with_mm": round(med_w, 2),
        "median_without_mm": round(med_o, 2),
        "delta_mm": round(delta, 2),
        "gate": {"pass_mm": GATE_PASS_MM, "ambiguous_mm": GATE_AMBIG_MM},
        "verdict": code, "next": nxt,
    }
    Path(a.out).expanduser().write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
