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

    # ⚠️ 2026-09-20 보강 (황도경 검토) — 초판은 세 가지가 "통과"로 세였다:
    #    (1) manifest 가 없으면 `exp_n is None` 이 True 라 조건이 참이 되어 ok 증가
    #    (2) 텐서가 0개여도 n=0 을 결과로 돌려줌 (형제 파일 export_deploy_ckpt 는 이미 고쳤다)
    #    (3) sha256_export 가 manifest 에 있는데 대조하지 않음 — 전송 무결성의 유일한 증거
    ema = sd.get("ema_model", {})
    tensors = [v for v in ema.values() if hasattr(v, "numel")]
    n = sum(int(v.numel()) for v in tensors)
    check("ema_model 에 텐서가 있다", len(tensors) > 0,
          f"텐서 {len(tensors)} / 항목 {len(ema)}  ← 0 이면 nParams 0 은 '작다'가 아니라 '없다'다")
    exp_n = (man or {}).get("nParams")
    if exp_n is None:
        check("nParams 가 manifest 와 일치", False,
              f"{n:,} / 기대 없음 — manifest 를 안 줬거나 nParams 키가 없다. "
              "대조 불가는 통과가 아니다 (--manifest 를 줘라)")
    else:
        check("nParams 가 manifest 와 일치", n == exp_n, f"{n:,} / 기대 {exp_n:,}")

    exp_sha = (man or {}).get("sha256_export")
    if exp_sha:
        import hashlib
        h = hashlib.sha256()
        with open(ckpt_path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        got_sha = h.hexdigest()
        check("sha256 가 manifest 와 일치", got_sha == exp_sha,
              f"{got_sha[:16]} / 기대 {exp_sha[:16]}")
    else:
        check("sha256 가 manifest 와 일치", False,
              "manifest 에 sha256_export 가 없다 — 전송 무결성을 증명할 수단이 없다")

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
        # ⚠️ 2026-09-21 — 학습 매니페스트를 그대로 넣었더니 KeyError 로 **죽었다**.
        #    죽는 검사기는 검사기가 아니다. 없는 키는 미판정으로 세고 계속 간다.
        exp, missing = manifest_contract(man)
        if missing:
            check("manifest 가 배포 계약 스키마인가", False,
                  f"없는 키 {missing} — 학습 매니페스트로 보인다. "
                  "export_deploy_ckpt.py 를 먼저 돌려라. 대조 불가는 통과가 아니다")
        # ⚠️ str() 비교라 둘 다 None 이면 'None' == 'None' 으로 일치가 된다.
        #    "둘 다 없음"이 "둘 다 맞음"으로 보이지 않게 None 을 따로 센다.
        both_none = [k for k in exp if got[k] is None and exp[k] is None] if exp else []
        bad = [k for k in exp if str(got[k]) != str(exp[k])] if exp else []
        check("계약값이 manifest 와 일치", bool(exp) and not bad and not both_none,
              f"{len(exp) - len(bad) - len(both_none)} / {len(CONTRACT_KEYS)}"
              + ("  대조할 키가 하나도 없다" if not exp else "")
              + (f"  불일치 {bad}" if bad else "")
              + (f"  양쪽 모두 None {both_none}" if both_none else ""))
        _ok, _why = anchor_verdict(man)
        check("청크 앵커 규약 chunk_start (D-AI-80)", _ok, _why)
    else:
        print("  [--] manifest 미지정 — 계약 대조 **미실행**")

    # ⚠️ 위 대조는 cfg 와 manifest 가 **서로 같은지**만 본다. 둘 다 'abs' 면 통과한다.
    #    이 파일 docstring 이 "relative 만 맞다"고 적어놓고 코드는 값을 검사하지 않았다.
    #    'rel' 은 소스 주석이 legacy buggy 라고 적은 별개 경로다 (2026-09-20, 황도경 검토).
    apr = at(cfg, "task.pose_repr.action_pose_repr")
    opr = at(cfg, "task.pose_repr.obs_pose_repr")
    check("action_pose_repr 가 'relative'", apr == "relative",
          f"{apr!r}  ← 'abs' 는 기본값이고 'rel' 은 legacy buggy 경로다. 셋 다 에러 없이 돈다")
    check("obs_pose_repr 가 'relative'", opr == "relative", f"{opr!r}")

    use_ema = at(cfg, "training.use_ema")
    check("training.use_ema 가 True", use_ema is True,
          f"{use_ema}  ← False 면 로더가 'model' 을 찾다 죽는다. ema_model 로 고정해서 읽어라")

    obs = at(cfg, "shape_meta.obs")
    names = sorted(obs.keys()) if hasattr(obs, "keys") else []
    print(f"  관측 키 {len(names)}개: {names}")
    return ok, total, {"cfg": cfg, "payload": p, "obs_names": names}


def _prepare_umi_path(umi_root: str | None) -> str:
    """Put the official UMI package on sys.path, reporting which route worked.
    공식 UMI 패키지를 sys.path 에 올리고 **어느 경로로 됐는지** 보고한다.

    `diffusion_policy` 는 UMI 저장소 안에 있다. 경로가 안 잡히면 hydra 가
    'Error locating target' 로 죽는데, 그건 체크포인트 문제가 아니라 환경 문제다.
    둘을 구분해서 찍는다."""
    import importlib

    # ⚠️ 스크립트를 경로로 실행하면 sys.path[0] 은 **스크립트 디렉터리**이고 cwd 가 아니다.
    #    `cd ~/handoff && python ~/S15P21A103/AI/tools/smoke_deploy_ckpt.py` 로 돌리면
    #    `umi_adapter` 도 `third_party` 도 안 보인다. cwd 를 먼저 올린다.
    for extra in (Path.cwd(), Path.cwd().parent):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    importlib.invalidate_caches()

    if importlib.util.find_spec("diffusion_policy") is not None:
        return "UMI 경로: 이미 import 가능"

    tried = []
    if umi_root:
        tried.append(Path(umi_root).expanduser())
    tried += [Path.cwd() / "third_party/umi",
              Path.cwd().parent / "third_party/umi",
              Path.home() / "handoff/third_party/umi"]

    try:
        from umi_adapter.upstream import enable
        enable()
        if importlib.util.find_spec("diffusion_policy") is not None:
            return "UMI 경로: umi_adapter.upstream.enable()"
    except Exception:                                  # noqa: BLE001
        pass

    for cand in tried:
        if (cand / "diffusion_policy").is_dir():
            sys.path.insert(0, str(cand))
            importlib.invalidate_caches()
            if importlib.util.find_spec("diffusion_policy") is not None:
                return f"UMI 경로: sys.path += {cand}"
    raise ImportError(
        f"diffusion_policy 를 못 찾았다.\n"
        f"    cwd        {Path.cwd()}\n"
        f"    sys.path[0:3] {sys.path[:3]}\n"
        f"    시도한 곳 {len(tried)}군데: {[str(t) for t in tried]}\n"
        "    --umi-root 로 UMI 저장소 경로(diffusion_policy 의 부모)를 주거나 공식 UMI 가 "
        "설치된 환경에서 돌려라. **체크포인트 문제가 아니라 환경 문제다.**")


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

    print(f"  {_prepare_umi_path(ctx.get('umi_root'))}")
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
    # ⚠️ 2026-09-21 — (8,10) 을 리터럴로 박아 현석 e120 ckpt(16,10)를 헛불합격시켰다.
    #    horizon 은 매니페스트에서 받는다. 못 받으면 **미판정**이지 통과가 아니다 (D-AI-81).
    _h = ((ctx or {}).get("manifest") or {}).get("actionSpec", {}).get("horizon")
    _d = ((ctx or {}).get("manifest") or {}).get("actionSpec", {}).get("dim") or 10
    if _h is None:
        check("출력 모양 (horizon, dim)", None,
              f"{act.shape} / 매니페스트에 actionSpec.horizon 이 없다 — 대조 불가는 통과가 아니다")
    else:
        check(f"출력 모양 ({_h}, {_d})", act.shape == (_h, _d),
              f"{act.shape} / 기대 ({_h}, {_d})")

    pos = np.abs(act[:, :3]).max()
    gap = act[:, 9]
    check("상대 위치가 cm 규모", pos < 0.5, f"최대 |dx,dy,dz| = {pos * 1000:.1f} mm")
    check("gap 이 0~0.09 안", float(gap.min()) >= -1e-3 and float(gap.max()) <= 0.091,
          f"[{gap.min():.4f}, {gap.max():.4f}] m")
    check("NaN/Inf 없음", bool(np.isfinite(act).all()))
    print(f"  action[0] = {np.round(act[0], 5)}")
    return ok, total


CONTRACT_KEYS = {
    "actionSpec.horizon": ("actionSpec", "horizon"),
    "actionSpec.n_action_steps": ("actionSpec", "n_action_steps"),
    "runtimeSpec.action_pose_repr": ("runtimeSpec", "required_kwarg", "action_pose_repr"),
    "runtimeSpec.obs_down_sample_steps": ("runtimeSpec", "obs_down_sample_steps"),
    "runtimeSpec.num_inference_steps": ("runtimeSpec", "num_inference_steps"),
}


def manifest_contract(man: dict) -> tuple[dict, list[str]]:
    """Read the deploy contract out of a manifest without raising on absent keys.
    없는 키에 터지지 않고 계약값을 읽는다. 반환 (읽힌 값, 없는 키 이름들)."""
    exp, missing = {}, []
    for name, path in CONTRACT_KEYS.items():
        cur = man
        for part in path:
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if cur is None:
            missing.append(name)
        else:
            exp[name] = cur
    return exp, missing


def anchor_verdict(man: dict | None) -> tuple[bool, str]:
    """Chunk-anchor gate: absence is NOT a pass.
    청크 앵커 규약 판정 (D-AI-80). 키가 없으면 '대조 불가'이지 통과가 아니다.

    매니페스트의 `compose: "T_next = T_cur @ A_relative"` 문자열은 청크 **안에서**
    누적하는지 앵커를 고정하는지 말해주지 않는다. 그 모호함으로 so101_infer.unroll 이
    누적으로 구현됐고 실물이 반경 565mm 원호를 돌았다. 이제 값으로 확인한다.
    """
    if not man:
        return False, "manifest 미지정 — 대조 불가. 통과가 아니다"
    ca = man.get("actionSpec", {}).get("chunk_anchor")
    if ca is None:
        return False, ("키 없음 — D-AI-80 이전 매니페스트다. 앵커/누적을 구분할 수 없다. "
                       "대조 불가는 통과가 아니다. export_deploy_ckpt 로 다시 뽑아라")
    if ca != "chunk_start":
        return False, f"chunk_anchor={ca!r} — 'chunk_start' 여야 한다"
    return True, "chunk_start"


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

    # [3-6] 청크 앵커 게이트 (D-AI-80). '없음'과 '괜찮음'이 갈리는지 본다
    good = {"actionSpec": {"chunk_anchor": "chunk_start"}}
    check("앵커 chunk_start -> 통과", anchor_verdict(good)[0] is True)
    check("앵커 키 없음 -> 실패 (판별행)", anchor_verdict(man)[0] is False,
          anchor_verdict(man)[1][:34])
    check("앵커 값이 누적 -> 실패 (판별행)",
          anchor_verdict({"actionSpec": {"chunk_anchor": "accumulate"}})[0] is False)
    check("manifest 없음 -> 실패 (판별행)", anchor_verdict(None)[0] is False)

    # [7-9] 학습 매니페스트(배포 계약 없음)를 넣어도 **죽지 않는지**. 2026-09-21 회귀
    train_man = {"run_id": "x", "best_checkpoint": {"epoch": 7}}
    _e, _m = manifest_contract(train_man)
    check("학습 매니페스트 -> 죽지 않고 미판정", _e == {} and len(_m) == len(CONTRACT_KEYS),
          f"읽힘 {len(_e)} / 없는 키 {len(_m)} / 전체 {len(CONTRACT_KEYS)}")
    _e2, _m2 = manifest_contract(man)
    check("배포 매니페스트 -> 전부 읽힘 (정답 아는 행)",
          len(_e2) == len(CONTRACT_KEYS) and _m2 == [],
          f"읽힘 {len(_e2)} / {len(CONTRACT_KEYS)}")
    _e3, _m3 = manifest_contract({"actionSpec": {"horizon": 16}})
    check("일부만 있는 매니페스트 -> 부분 보고 (판별행)",
          len(_e3) == 1 and len(_m3) == len(CONTRACT_KEYS) - 1,
          f"읽힘 {len(_e3)} / 없음 {len(_m3)}")
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
    ap.add_argument("--umi-root", help="공식 UMI 저장소 경로 (diffusion_policy 의 부모)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.checkpoint:
        ap.error("--checkpoint 가 필요하다 (또는 --selftest)")

    man = json.loads(Path(a.manifest).read_text(encoding="utf-8")) if a.manifest else None
    print("=== 단계 0 · 파일 온전성과 계약 대조 (torch 만 필요) ===")
    o0, t0, ctx = stage0(Path(a.checkpoint).expanduser(), man)
    ctx["umi_root"] = a.umi_root
    ctx["manifest"] = man        # 단계 1 이 horizon 을 여기서 받는다 (D-AI-81)
    print(f"  단계 0: {o0} / {t0}")

    o1 = t1 = 0
    skipped = None
    failed = None
    if a.skip_stage1:
        skipped = "--skip-stage1"
    else:
        print("\n=== 단계 1 · 실제 추론 1회 (hydra + diffusion_policy 필요) ===")
        try:
            o1, t1 = stage1(ctx)
            print(f"  단계 1: {o1} / {t1}")
        except ImportError as exc:
            skipped = f"의존성 없음: {exc}"
        # ⚠️ 2026-09-20 (황도경 검토) — 초판은 ImportError 와 그 외 **전부**를 같은
        #    "미실행" 통에 넣고 "단계 0 이 전부 OK 면 파일은 정상이다" 라고 안내했다.
        #    state_dict 키·shape 불일치(RuntimeError)는 **이 도구가 잡으라고 만든 결함**인데,
        #    받는 쪽은 자기 환경 탓으로 읽는다. 환경 문제와 체크포인트 문제를 가른다.
        except Exception as exc:                       # noqa: BLE001 — 사유를 남긴다
            failed = f"{type(exc).__name__}: {exc}"

    print(f"\n합계 {o0 + o1} / {t0 + t1}")
    if failed:
        print(f"!! 단계 1 **실패** — {failed}")
        print("   이건 환경 문제가 아니라 체크포인트 문제일 가능성이 높다.")
        print("   (환경 문제라면 ImportError 로 나오고 '미실행' 로 보고된다)")
        sys.exit(1)
    if skipped:
        print(f"⚠️ 단계 1 **미실행** — {skipped}")
        print("   단계 0 이 전부 OK 여도 '파일 정상' 이 아니다 — 추론을 한 번도 안 돌렸다.")
        print("   추론 환경에서 다시 돌려라.")
        sys.exit(2 if o0 == t0 else 1)
    sys.exit(0 if (o0 + o1) == (t0 + t1) else 1)


if __name__ == "__main__":
    main()
