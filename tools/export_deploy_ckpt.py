"""Strip a UMI training checkpoint down to a deployable payload + contract manifest.
UMI 학습 체크포인트를 배포용 payload 와 계약 manifest 로 줄인다.

왜 필요한가 (2026-09-19)
------------------------
학습 ckpt 는 AdamW 2모멘트와 non-EMA 가중치를 함께 들고 있어 두 배 이상 크다.

    A 조건 ckpt   152,842,531 바이트
    B 조건 ckpt   305,646,454 바이트      차이 = optimizer 상태
    배포에 필요한 것  ema_model 만        약 76MB

추론 쪽(Jetson/HW)에 넘기는 것은 **ema_model + cfg + 계약 manifest** 셋이다.
manifest 가 없으면 받는 쪽이 텐서 의미를 추측하게 되고, 그 추측은 에러 없이 틀린다.

계약 값의 출처 — 전부 실측·소스 대조다 🟢
------------------------------------------
    MEASURE_action_convention_0918.md   layout · rot6d 행 규약 · 곱 순서 · execSlice
    MEASURE_ckpt_contract_0918.md       nParams · lr · num_inference_steps · horizon
    MEASURE_folds_real_0919.md          성능 (홀드아웃 궤적 오차. 롤아웃 아님)

⚠️ **조용히 틀리는 경로 셋** 을 manifest 에 박는다.
`get_real_umi_action(action, obs, action_pose_repr=...)` 의 기본값이 `'abs'` 다.
인자를 안 넘기면 상대를 절대로 해석해 팔이 원점으로 간다. `'rel'` 은 소스 주석이
"legacy buggy implementation" 이라고 적은 별개 경로다. **`'relative'` 만 맞다.**
셋 다 에러 없이 돈다.

Usage
-----
  python export_deploy_ckpt.py --selftest
  python export_deploy_ckpt.py --checkpoint ~/handoff/outputs/f0918_46_B/checkpoints/latest.ckpt \
      --out ~/handoff/outputs/deploy/so101_pick_v1.ckpt --note "5-fold B, fold 46"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# ckpt payload 에서 기대하는 최상위 키
EXPECTED_TOP = ("cfg", "state_dicts")
KEEP_STATE = ("ema_model",)
DROP_STATE = ("model", "optimizer")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def inspect(payload: dict) -> dict:
    """Report what the payload has, with denominators. 모수와 함께 무엇이 있는지 낸다."""
    top = list(payload.keys())
    sd = payload.get("state_dicts", {})
    sd_keys = list(sd.keys()) if isinstance(sd, dict) else []
    missing_top = [k for k in EXPECTED_TOP if k not in top]
    missing_state = [k for k in KEEP_STATE if k not in sd_keys]
    return {
        "top_keys": top,
        "top_found": f"{len(EXPECTED_TOP) - len(missing_top)} / {len(EXPECTED_TOP)}",
        "missing_top": missing_top,
        "state_dict_keys": sd_keys,
        "state_found": f"{len(KEEP_STATE) - len(missing_state)} / {len(KEEP_STATE)}",
        "missing_state": missing_state,
        "will_drop": [k for k in DROP_STATE if k in sd_keys],
    }


def count_params(state: dict) -> int:
    """Total element count of a state dict, refusing to report a silent zero.
    상태사전 전체 원소 수. 조용한 0 을 보고하지 않는다.

    ⚠️ 2026-09-19 정정 — 초판은 `except AttributeError: pass` 였다. 모든 값이
    텐서가 아니어도 0 을 그냥 돌려줬다. "없음"과 "괜찮음"이 같은 출력이다.
    이제 모수를 같이 세고, 센 항목이 0 이면 예외로 죽는다."""
    n = counted = 0
    total = len(state)
    for v in state.values():
        numel = getattr(v, "numel", None)
        if callable(numel):
            n += int(numel())
            counted += 1
    if counted == 0:
        raise ValueError(f"!! 텐서인 항목이 0 / {total} 이다. nParams 를 내지 않는다")
    if counted < total:
        print(f"   (주의) 텐서 {counted} / 전체 {total} 항목만 셌다")
    return n


def plain(v):
    """Coerce OmegaConf / exotic containers to plain JSON-safe Python.
    OmegaConf 등 특수 컨테이너를 JSON 으로 나갈 수 있는 순수 파이썬으로 바꾼다.

    ⚠️ 2026-09-19: cfg 값이 `ListConfig` 라 `json.dumps` 가 죽었다. 자체검증이
    직렬화를 한 번도 시도하지 않아 못 잡았다. 그 행을 추가했다."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(v):
            return OmegaConf.to_container(v, resolve=True)
    except ImportError:
        # omegaconf 가 없는 환경(수신측 스모크)에서는 아래 일반 경로로 내려간다.
        # 이건 "못 찾았다"가 아니라 "이 환경엔 원래 없다"라서 조용해도 된다.
        pass
    if hasattr(v, "items"):
        return {str(k): plain(x) for k, x in v.items()}
    if hasattr(v, "__iter__"):
        return [plain(x) for x in v]
    return str(v)


def at(node, path: str, default=None):
    """Read one EXPLICIT dotted path, returning JSON-safe Python.
    점 경로 하나를 명시적으로 읽어 JSON 안전한 값으로 돌려준다.

    ⚠️ 2026-09-19: 초판은 키 이름을 재귀 탐색했다. cfg 안에 `horizon` 이 수십 번 나오고
    (obs 는 2, action 은 8) **먼저 찾히는 것**을 집었다. 8 이 나온 건 운이다.
    이름이 겹치는 설정에서 재귀 탐색은 판별력이 없다. 경로를 적는다."""
    cur = node
    for part in path.split("."):
        if not hasattr(cur, "__getitem__"):
            return default
        try:
            cur = cur[part]
        except Exception:                              # noqa: BLE001
            return default
    return plain(cur)


def obs_history(cfg) -> tuple[object, str]:
    """Observation history length, with the denominator. 관측 이력 길이와 모수.

    `n_obs_steps` 라는 키는 이 체크포인트에 **없다**. 항목마다
    `shape_meta.obs.<key>.horizon` 으로 들어 있고 전부 같아야 한다."""
    obs = at(cfg, "shape_meta.obs")
    if obs is None or not hasattr(obs, "items"):
        return None, "shape_meta.obs 없음"
    vals = {k: at(v, "horizon") for k, v in obs.items()}
    uniq = sorted({v for v in vals.values() if v is not None})
    detail = f"{len(vals)}개 obs 키 중 값 있는 것 {len([v for v in vals.values() if v is not None])}, 서로 다른 값 {uniq}"
    if len(uniq) == 1:
        return uniq[0], detail
    return None, detail + "  ← 값이 갈린다. 하나로 못 적는다"


def build_manifest(cfg, n_params: int, src: Path, out: Path,
                   note: str, epoch) -> dict:
    """The contract the receiving side must not have to guess.
    받는 쪽이 추측하지 않아도 되도록 적는 계약."""
    n_obs, n_obs_detail = obs_history(cfg)
    return {
        "contractVersion": "UNSET — Jetson 추론 코드와 합의 후 채운다",
        "source_checkpoint": str(src),
        "exported_to": out.name,
        "note": note,
        "epoch": epoch,
        "nParams": n_params,
        "actionSpec": {
            "dim": at(cfg, "policy.obs_encoder.shape_meta.action.shape"),
            "horizon": at(cfg, "shape_meta.action.horizon"),
            "n_action_steps": at(cfg, "n_action_steps"),
            "rateHz": 10,
            "rateHz_note": "공칭이다. v10 원본 간격이 불균일하므로 0.1초 고정 가정 금지",
            "layout": {"0:3": "dx,dy,dz (m, 상대)",
                       "3:9": "rot6d",
                       "9": "gap (m, 절대)"},
            "rotation": "회전행렬의 첫 두 **행**. 열이 아니다",
            "rotation_rep_cfg": at(cfg, "policy.obs_encoder.shape_meta.action.rotation_rep"),
            "compose": "T_next = T_cur @ A_relative",
            "anchor": "T_cur = 현재 TCP = 패드 사이 중심 (손끝 아님)",
            "gapUnit": "m",
            "gapRange": [0.0, 0.09],
            "execSlice": [1, 5],
            "execSlice_note": ("cfg 값이 아니다. 모델은 8점을 내고(n_action_steps) "
                               "evaluate.py 가 action_steps=4 로 index [1,5) 만 실행한 뒤 "
                               "재관측한다. 실행측 선택이므로 바꾸려면 여기도 바꾼다"),
            "converted_form": "7차원 축각 절대 (pos3 + rotvec3 + gap)",
        },
        "runtimeSpec": {
            "decoder": "diffusion_policy.common.replay_buffer / real_inference_util.get_real_umi_action",
            "required_kwarg": {
                "action_pose_repr": at(cfg, "task.pose_repr.action_pose_repr")},
            "obs_pose_repr": at(cfg, "task.pose_repr.obs_pose_repr"),
            "WARNING": ("기본값이 'abs' 다. 인자를 안 넘기면 상대를 절대로 해석한다. "
                        "'rel' 은 소스 주석이 legacy buggy implementation 이라 적은 별개 경로다. "
                        "'relative' 만 맞고 셋 다 에러 없이 돈다"),
            "obs_history_steps": n_obs,
            "obs_history_source": ("shape_meta.obs.<key>.horizon — `n_obs_steps` 라는 키는 "
                                   f"이 체크포인트에 없다. 집계: {n_obs_detail}"),
            "obs_keys": sorted((at(cfg, "shape_meta.obs") or {}).keys()) or None,
            "img_obs_horizon": at(cfg, "task.img_obs_horizon"),
            "low_dim_obs_horizon": at(cfg, "task.low_dim_obs_horizon"),
            "obs_down_sample_steps": at(cfg, "task.obs_down_sample_steps"),
            "num_inference_steps": at(cfg, "policy.num_inference_steps"),
            "vision_encoder": at(cfg, "policy.obs_encoder.model_name"),
            "camera_obs_latency_s": at(cfg, "task.camera_obs_latency"),
            "frameworkVersion_train": "torch 2.13.0+cu126 (V100, sm_70)",
            "frameworkVersion_infer": "UNSET — Jetson 보드 확정 후",
        },
        "robotSpec": {
            "pinch_offset_local_m": "ver1 [0, 0, -0.158118819] (wrist_roll_link 기준)",
            "verified": ("handoff 시뮬 모델이 이미 ver1 기하다. policy_to_joints.py --probe-tcp "
                         "실측 z = -0.15811881850916723, ver1 기대값과 4.9e-10 m 차이"),
            "WARNING": ("AI 저장소 구 MJCF 경로(so101.yaml -0.080)로 IK 를 풀면 접근축으로 "
                        "78.118819mm 상수 편향. 같은 프레임 확인됨(MJCF gripper body vs URDF "
                        "wrist_roll_link 회전 상대각 0.0003도). 파지 여유는 ±25mm 뿐이고 "
                        "이런 상수 편향은 학습이 지우지 못한다"),
            "jaw_axis": ("ver1 jaw 는 레거시(+X) 기준 gripper 로컬 +Z 주위 +92.7889도. "
                         "URDF 패드 기하에서 검산(문서값과 0.0011도 차이). 현행 IK 는 jaw 를 "
                         "구속하지 않으므로 wrist_roll 로 보정해야 한다"),
            "handEye_T_camera_pinch": "AI/configs/real/umi_s22_canonical_pinch_side_grasp.json",
            "handEye_scope": "시연 리그 기준이다. 로봇팔 카메라 마운트 자세는 미상 (URDF 카메라 링크 0개)",
        },
        "performance": {
            "metric": "홀드아웃 상대 action chunk 예측 오차. **롤아웃 성공률이 아니다**",
            "source": "MEASURE_folds_real_0919.md (74편 축수정본, 5-fold)",
            "identity_relative_improvement": {"trans": "67.4~70.4%",
                                              "rot": "40.6~43.7%",
                                              "gap": "32.4~41.5%"},
            "rollout_success_rate": "미측정. 실물 평가 0건",
        },
    }


def export(src: Path, out: Path, note: str) -> int:
    """Strip to ema_model + write the contract manifest, then read back to verify.
    ema_model 만 남기고 계약 manifest 를 쓴 뒤, 되읽어 검산한다."""
    import torch
    payload = torch.load(src, map_location="cpu", weights_only=False)
    info = inspect(payload)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if info["missing_top"] or info["missing_state"]:
        print(f"!! 필수 키 누락 {info['missing_top'] + info['missing_state']} — 중단")
        return 1

    sd = payload["state_dicts"]
    ema = sd["ema_model"]
    n_params = count_params(ema)
    try:
        import dill
        epoch = dill.loads(payload["pickles"]["epoch"])
    except Exception:                                  # noqa: BLE001 — 없으면 없는 대로 적는다
        epoch = "미기록"

    slim = {"cfg": payload["cfg"], "state_dicts": {"ema_model": ema}}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out)

    man = build_manifest(payload["cfg"], n_params, src, out, note, epoch)
    man["sha256_source"] = sha256(src)
    man["sha256_export"] = sha256(out)
    man["bytes_source"] = src.stat().st_size
    man["bytes_export"] = out.stat().st_size
    mpath = out.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nnParams(ema)  {n_params:,}")
    print(f"원본          {man['bytes_source']:,} 바이트")
    print(f"배포본        {man['bytes_export']:,} 바이트  "
          f"({man['bytes_export'] / man['bytes_source'] * 100:.1f}%)")
    print(f"버린 것       {info['will_drop']}")
    print(f"\n-> {out}\n-> {mpath}")

    # 되읽기 검산 — 저장했다고 담긴 게 아니다
    back = torch.load(out, map_location="cpu", weights_only=False)
    ok = (set(back["state_dicts"]) == {"ema_model"}
          and count_params(back["state_dicts"]["ema_model"]) == n_params)
    print(f"되읽기 검산   {'OK' if ok else '!! 불일치'}  "
          f"(키 {set(back['state_dicts'])}, nParams {count_params(back['state_dicts']['ema_model']):,})")

    # 계약 빈칸 검사 — 받는 쪽이 추측하게 되는 항목을 모수와 함께 찍는다
    must = {
        "actionSpec.dim": man["actionSpec"]["dim"],
        "actionSpec.horizon": man["actionSpec"]["horizon"],
        "actionSpec.n_action_steps": man["actionSpec"]["n_action_steps"],
        "runtimeSpec.action_pose_repr": man["runtimeSpec"]["required_kwarg"]["action_pose_repr"],
        "runtimeSpec.obs_history_steps": man["runtimeSpec"]["obs_history_steps"],
        "runtimeSpec.obs_down_sample_steps": man["runtimeSpec"]["obs_down_sample_steps"],
        "runtimeSpec.num_inference_steps": man["runtimeSpec"]["num_inference_steps"],
    }
    blank = [k for k, v in must.items() if v is None]
    print(f"계약 필수항목  채움 {len(must) - len(blank)} / {len(must)}")
    for k, v in must.items():
        print(f"  {'OK ' if v is not None else '!! '} {k:<36} {v}")
    if blank:
        print(f"\n!! 빈칸 {len(blank)}개 — 받는 쪽이 추측하게 된다. 보내기 전에 채워라.")
        return 1

    # ⚠️ 2026-09-20 (황도경 검토) — 빈칸 검사는 `is None` 만 봤다. cfg 의 pose_repr 이
    #    'abs' 나 'rel'(소스가 legacy buggy 라 적은 경로)이어도 **값이 있으므로 초록불**이고,
    #    manifest 는 받는 쪽에 그 값을 쓰라고 지시한다. 이 파일 docstring 이 경고한
    #    바로 그 경로를 계약서에 박아 내보내는 셈이다. 값 자체를 검사한다.
    #    수신측(run_policy_realtime)이 이미 != "relative" 로 막는다 —
    #    **내보내는 쪽이 받는 쪽보다 느슨하면 안 된다.**
    wrong = {k: v for k, v in (
        ("runtimeSpec.action_pose_repr", man["runtimeSpec"]["required_kwarg"]["action_pose_repr"]),
        ("runtimeSpec.obs_pose_repr", man["runtimeSpec"]["obs_pose_repr"]),
    ) if v != "relative"}
    if wrong:
        print(f"\n!! pose_repr 이 'relative' 가 아니다: {wrong}")
        print("   'abs' 는 기본값이고 'rel' 은 legacy buggy 경로다. 셋 다 에러 없이 돈다.")
        print("   이 체크포인트는 내보내지 않는다 — 학습 cfg 부터 확인해라.")
        return 1
    print(f"  OK  pose_repr 양쪽 모두 'relative'")
    return 0 if ok else 1


def selftest() -> int:
    """Known-answer rows on synthetic payloads, including deliberately broken ones.
    합성 payload 로 정답 아는 행. 고의로 망가뜨린 입력을 포함한다."""
    ok = total = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok, total
        total += 1
        ok += bool(cond)
        print(f"[{total}] {name:<44} {'OK' if cond else '!! 실패'}  {detail}")

    good = {"cfg": {}, "state_dicts": {"model": {}, "ema_model": {}, "optimizer": {}}}
    i = inspect(good)
    check("정상 payload 인식", i["missing_top"] == [] and i["missing_state"] == [],
          f"top {i['top_found']} · state {i['state_found']}")
    check("버릴 키 식별", set(i["will_drop"]) == {"model", "optimizer"}, str(i["will_drop"]))
    check("ema_model 없음 -> 누락", inspect({"cfg": {}, "state_dicts": {"model": {}}})["missing_state"] == ["ema_model"])
    check("cfg 없음 -> 누락", inspect({"state_dicts": {"ema_model": {}}})["missing_top"] == ["cfg"])
    e = inspect({})
    check("빈 payload -> 전부 누락", len(e["missing_top"]) == 2 and e["top_found"] == "0 / 2", e["top_found"])

    # 이름이 겹치는 cfg. obs.horizon=2 와 action.horizon=8 이 공존한다
    cfg = {
        "n_action_steps": 8,
        "shape_meta": {
            "obs": {"camera0_rgb": {"horizon": 2}, "robot0_eef_pos": {"horizon": 2},
                    "robot0_gripper_width": {"horizon": 2}},
            "action": {"horizon": 8},
        },
        "policy": {"num_inference_steps": 16,
                   "obs_encoder": {"model_name": "resnet18",
                                   "shape_meta": {"action": {"shape": [10],
                                                             "rotation_rep": "rotation_6d"}}}},
        "task": {"pose_repr": {"action_pose_repr": "relative", "obs_pose_repr": "relative"},
                 "obs_down_sample_steps": 1, "img_obs_horizon": 2, "low_dim_obs_horizon": 2,
                 "camera_obs_latency": 0.125},
    }
    man = build_manifest(cfg, 19_078_252, Path("/x/a.ckpt"), Path("/y/b.ckpt"), "t", 59)
    a, r = man["actionSpec"], man["runtimeSpec"]
    check("이름 겹침: action horizon 8 을 집는다", a["horizon"] == 8, str(a["horizon"]))
    check("이름 겹침: obs 이력은 2 로 따로", r["obs_history_steps"] == 2, str(r["obs_history_steps"]))
    check("obs 이력 모수 표기", "3개 obs 키" in r["obs_history_source"])
    check("dim 은 shape_meta.action.shape", a["dim"] == [10])
    check("n_action_steps 8 · execSlice 실행측",
          a["n_action_steps"] == 8 and "cfg 값이 아니다" in a["execSlice_note"])
    check("pose_repr 명시 경로", r["required_kwarg"]["action_pose_repr"] == "relative")
    check("obs_down_sample_steps 1", r["obs_down_sample_steps"] == 1)
    check("num_inference_steps 16", r["num_inference_steps"] == 16)
    check("rot6d 행 규약 명시", "행" in a["rotation"])
    check("곱 순서 명시", a["compose"] == "T_next = T_cur @ A_relative")
    check("ver1 78.1mm 경고", "78.118819mm" in man["robotSpec"]["WARNING"])
    check("롤아웃 아님 명시", "롤아웃 성공률이 아니다" in man["performance"]["metric"])
    check("contractVersion 미정 표기", man["contractVersion"].startswith("UNSET"))

    bad = dict(cfg)
    bad["shape_meta"] = {"obs": {"a": {"horizon": 2}, "b": {"horizon": 3}}, "action": {"horizon": 8}}
    m2 = build_manifest(bad, 0, Path("/x"), Path("/y"), "", None)
    check("obs horizon 불일치 -> None + 사유",
          m2["runtimeSpec"]["obs_history_steps"] is None
          and "갈린다" in m2["runtimeSpec"]["obs_history_source"])

    # 직렬화 — 이게 없어서 ListConfig 를 못 잡았다
    class FakeSeq:                                     # list 를 상속하지 않는 시퀀스
        def __init__(self, *v):
            self._v = list(v)
        def __iter__(self):
            return iter(self._v)
        def __len__(self):
            return len(self._v)
        def __getitem__(self, i):
            return self._v[i]

    exotic = {k: v for k, v in cfg.items()}
    exotic["policy"] = {"num_inference_steps": 16,
                        "obs_encoder": {"model_name": "resnet18",
                                        "shape_meta": {"action": {"shape": FakeSeq(10),
                                                                  "rotation_rep": "rotation_6d"}}}}
    m4 = build_manifest(exotic, 0, Path("/x"), Path("/y"), "", None)
    check("특수 시퀀스 -> 순수 list", m4["actionSpec"]["dim"] == [10], str(m4["actionSpec"]["dim"]))
    try:
        json.dumps(m4, ensure_ascii=False)
        ser = True
    except TypeError:
        ser = False
    check("manifest 가 json.dumps 된다", ser)
    try:
        json.dumps({"x": FakeSeq(1)})
        disc = False
    except TypeError:
        disc = True
    check("판별력: 변환 안 하면 json 이 죽는다", disc)

    class FakeT:
        def numel(self): return 7

    check("nParams: 텐서만 세고 합이 맞는다",
          count_params({"a": FakeT(), "b": FakeT()}) == 14)
    check("nParams: 텐서 반 · 비텐서 반이어도 합이 맞는다",
          count_params({"a": FakeT(), "b": "문자열"}) == 7)
    try:
        count_params({"a": "문자열", "b": 3})
        zero_guard = False
    except ValueError:
        zero_guard = True
    check("판별력: 텐서가 0개면 0 을 돌려주지 않고 죽는다", zero_guard)

    m3 = build_manifest({}, 0, Path("/x"), Path("/y"), "", None)
    check("빈 cfg -> 전부 None",
          m3["actionSpec"]["horizon"] is None and m3["runtimeSpec"]["obs_history_steps"] is None)

    try:
        import torch  # noqa: F401
        torch_ok = True
    except ImportError:
        torch_ok = False

    print(f"\n자체검증 {ok} / {total}")
    if not torch_ok:
        print("⚠️ torch 없음 — 저장·되읽기 경로는 **미실행**이다. 통과가 아니다.")
        return 2
    return 0 if ok == total else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--out")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not (a.checkpoint and a.out):
        ap.error("--checkpoint 와 --out 이 필요하다 (또는 --selftest)")
    sys.exit(export(Path(a.checkpoint).expanduser(), Path(a.out).expanduser(), a.note))


if __name__ == "__main__":
    main()
