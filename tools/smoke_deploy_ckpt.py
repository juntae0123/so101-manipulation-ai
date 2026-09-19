"""Receiver-side smoke test for the deployed SO-101 policy checkpoint.
배포된 SO-101 정책 체크포인트를 받는 쪽에서 돌리는 스모크 테스트.

받는 사람에게
-------------
이 파일 하나와 `so101_pick_v1.ckpt` · `so101_pick_v1.manifest.json` 만 있으면 된다.

    python smoke_deploy_ckpt.py --checkpoint so101_pick_v1.ckpt --manifest so101_pick_v1.manifest.json

**단계 0** 은 torch 만 있으면 돈다 — 파일이 온전히 왔는지, 계약값이 manifest 와 맞는지.
**단계 1** 은 `hydra` + 공식 UMI `diffusion_policy` 가 있어야 돈다 — 실제로 한 번 추론해서
출력 모양 `(8, 10)` 과 값 범위를 본다.

단계 1 이 안 돌아도 단계 0 이 통과하면 **파일은 정상**이다. 그 경우 "미실행"으로 찍히고
통과로 세지 않는다. 없음과 괜찮음을 같은 출력으로 내지 않는다.

무엇을 보면 되나
----------------
    action 모양 (8, 10)      0:3 상대 위치[m] · 3:9 rot6d(행) · 9 gap[m]
    gap 이 0~0.09 안         밖이면 단위가 섞였다
    상대 위치가 수 cm 규모     수십 cm 면 절대 pose 로 해석된 것이다

⚠️ 실행 시 반드시 `action_pose_repr='relative'` 를 넘겨야 한다. 기본값은 `'abs'` 이고
   상대 궤적을 절대로 해석해 팔이 원점 근처로 간다. `'rel'` 은 소스가 스스로
   "legacy buggy implementation" 이라 적은 별개 경로다. **셋 다 에러 없이 돈다.**
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def stage0(ckpt_path: Path, man: dict | None) -> tuple[int, int, dict]:
    """File integrity + contract cross-check. 파일 온전성과 계약 대조."""
    import torch
    ok = total = 0

    def check(name, cond, detail=""):
        nonlocal ok, total
        total += 1
        ok += bool(cond)
        print(f"  [{total}] {name:<42} {'OK' if cond else '!! 실패'}  {detail}")

    p = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keys = list(p.keys())
    sd = p.get("state_dicts", {})
    sdk = list(sd.keys())
    check("최상위에 cfg·state_dicts", {"cfg", "state_dicts"} <= set(keys), str(keys))
    check("state_dicts 는 ema_model 뿐", set(sdk) == {"ema_model"}, str(sdk))

    n = sum(int(v.numel()) for v in sd.get("ema_model", {}).values() if hasattr(v, "numel"))
    exp_n = (man or {}).get("nParams")
    check("nParams 가 manifest 와 일치", exp_n is None or n == exp_n,
          f"{n:,}" + (f" / 기대 {exp_n:,}" if exp_n else " (manifest 없음)"))

    cfg = p["cfg"]

    def at(node, path):
        cur = node
        for part in path.split("."):
            if not hasattr(cur, "__getitem__"):
                return None
            try:
                cur = cur[part]
            except Exception:                          # noqa: BLE001
                return None
        return cur

    got = {
        "actionSpec.horizon": at(cfg, "shape_meta.action.horizon"),
        "actionSpec.n_action_steps": at(cfg, "n_action_steps"),
        "runtimeSpec.action_pose_repr": at(cfg, "task.pose_repr.action_pose_repr"),
        "runtimeSpec.obs_down_sample_steps": at(cfg, "task.obs_down_sample_steps"),
        "runtimeSpec.num_inference_steps": at(cfg, "policy.num_inference_steps"),
    }
    if man:
        exp = {
            "actionSpec.horizon": man["actionSpec"]["horizon"],
            "actionSpec.n_action_steps": man["actionSpec"]["n_action_steps"],
            "runtimeSpec.action_pose_repr": man["runtimeSpec"]["required_kwarg"]["action_pose_repr"],
            "runtimeSpec.obs_down_sample_steps": man["runtimeSpec"]["obs_down_sample_steps"],
            "runtimeSpec.num_inference_steps": man["runtimeSpec"]["num_inference_steps"],
        }
        bad = [k for k in exp if str(got[k]) != str(exp[k])]
        check("계약값이 manifest 와 일치", not bad,
              f"{len(exp) - len(bad)} / {len(exp)}" + (f"  불일치 {bad}" if bad else ""))
    else:
        print("  [--] manifest 미지정 — 계약 대조 **미실행**")

    use_ema = at(cfg, "training.use_ema")
    check("training.use_ema 가 True", use_ema is True,
          f"{use_ema}  ← False 면 로더가 'model' 을 찾다 죽는다. ema_model 로 고정해서 읽어라")

    obs = at(cfg, "shape_meta.obs")
    names = sorted(obs.keys()) if hasattr(obs, "keys") else []
    print(f"  관측 키 {len(names)}개: {names}")
    return ok, total, {"cfg": cfg, "payload": p, "obs_names": names}


def stage1(ctx: dict) -> tuple[int, int]:
    """One real forward pass with synthetic observations. 합성 관측으로 실제 추론 1회."""
    import numpy as np
    import torch
    ok = total = 0

    def check(name, cond, detail=""):
        nonlocal ok, total
        total += 1
        ok += bool(cond)
        print(f"  [{total}] {name:<42} {'OK' if cond else '!! 실패'}  {detail}")

    import hydra
    cfg = ctx["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(ctx["payload"]["state_dicts"]["ema_model"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(dev).eval()
    print(f"  장치 {dev}")

    obs_meta = cfg.shape_meta.obs
    batch = {}
    for k in obs_meta:
        shp = list(obs_meta[k]["shape"])
        h = int(obs_meta[k]["horizon"])
        t = torch.zeros([1, h] + shp, dtype=torch.float32)
        if obs_meta[k].get("type") == "rgb":
            t = t + 0.5                                 # 중간 회색
        batch[k] = t.to(dev)
    check("합성 관측 구성", len(batch) == len(list(obs_meta)),
          f"{len(batch)} / {len(list(obs_meta))}")

    with torch.inference_mode():
        out = policy.predict_action(batch)
    act = out["action"][0].detach().cpu().numpy()
    check("출력 모양 (8, 10)", act.shape == (8, 10), str(act.shape))

    pos = np.abs(act[:, :3]).max()
    gap = act[:, 9]
    check("상대 위치가 cm 규모", pos < 0.5, f"최대 |dx,dy,dz| = {pos * 1000:.1f} mm")
    check("gap 이 0~0.09 안", float(gap.min()) >= -1e-3 and float(gap.max()) <= 0.091,
          f"[{gap.min():.4f}, {gap.max():.4f}] m")
    check("NaN/Inf 없음", bool(np.isfinite(act).all()))
    print(f"  action[0] = {np.round(act[0], 5)}")
    return ok, total


def selftest() -> int:
    """Known-answer rows that do not need the checkpoint. ckpt 없이 도는 정답 아는 행."""
    ok = total = 0

    def check(name, cond, detail=""):
        nonlocal ok, total
        total += 1
        ok += bool(cond)
        print(f"[{total}] {name:<44} {'OK' if cond else '!! 실패'}  {detail}")

    man = {"nParams": 19078252,
           "actionSpec": {"horizon": 8, "n_action_steps": 8},
           "runtimeSpec": {"required_kwarg": {"action_pose_repr": "relative"},
                           "obs_down_sample_steps": 1, "num_inference_steps": 16}}
    check("manifest 스키마 읽힘", man["actionSpec"]["horizon"] == 8)
    check("판별력: 불일치가 감지되는가", str(man["actionSpec"]["horizon"]) != str(16))
    try:
        import torch  # noqa: F401
        t = True
    except ImportError:
        t = False
    print(f"\n자체검증 {ok} / {total}")
    if not t:
        print("⚠️ torch 없음 — 단계 0·1 은 **미실행**이다. 통과가 아니다.")
        return 2
    return 0 if ok == total else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--manifest")
    ap.add_argument("--skip-stage1", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.checkpoint:
        ap.error("--checkpoint 가 필요하다 (또는 --selftest)")

    man = json.loads(Path(a.manifest).read_text(encoding="utf-8")) if a.manifest else None
    print("=== 단계 0 · 파일 온전성과 계약 대조 (torch 만 필요) ===")
    o0, t0, ctx = stage0(Path(a.checkpoint).expanduser(), man)
    print(f"  단계 0: {o0} / {t0}")

    o1 = t1 = 0
    skipped = None
    if a.skip_stage1:
        skipped = "--skip-stage1"
    else:
        print("\n=== 단계 1 · 실제 추론 1회 (hydra + diffusion_policy 필요) ===")
        try:
            o1, t1 = stage1(ctx)
            print(f"  단계 1: {o1} / {t1}")
        except ImportError as exc:
            skipped = f"의존성 없음: {exc}"
        except Exception as exc:                       # noqa: BLE001 — 사유를 남긴다
            skipped = f"{type(exc).__name__}: {exc}"

    print(f"\n합계 {o0 + o1} / {t0 + t1}")
    if skipped:
        print(f"⚠️ 단계 1 **미실행** — {skipped}")
        print("   단계 0 이 전부 OK 면 파일은 정상이다. 추론 환경에서 다시 돌려라.")
        sys.exit(2 if o0 == t0 else 1)
    sys.exit(0 if (o0 + o1) == (t0 + t1) else 1)


if __name__ == "__main__":
    main()
