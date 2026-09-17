"""Separate sampling noise from systematic bias in diffusion action predictions.
정책 예측 오차가 '표집 잡음' 인지 '계통 편향' 인지 가르는 계측기.

왜 필요한가 (2026-09-18)
------------------------
E2 A조건(실데이터만) 60 epoch 결과:

    ds_real_holdout   trans 40.99mm (identity 50.72 · +19.2%)   rot 5.63도 (identity 2.45 · -129.5%)
    ds_real_train     trans 29.18mm (identity 49.22 · +40.7%)   rot 4.57도 (identity 2.31 · -98.4%)

병진·gap 은 **학습 데이터에서도 홀드아웃에서도** identity 를 이긴다 → 학습은 됐다.
그런데 회전만 **학습 데이터에서조차** identity 보다 나쁘다. 자기가 본 데이터를 못 맞히는
모델은 없으므로, 회전 채널은 둘 중 하나다.

    (a) 표집 잡음  — 실 시연의 0.8초 회전량이 2.4도뿐이라 diffusion 표집 잔차에 묻힌다.
                     → 같은 관측을 K번 표집하면 결과가 매번 다르고, 평균은 오차가 준다.
                     → **실제 결과다.** 표현/표집 스텝 문제이지 버그가 아니다.
    (b) 계통 편향  — 규약(행/열), 정규화 역변환, 시간 정렬 중 하나가 틀렸다.
                     → K번 표집해도 결과가 같고, 평균을 내도 오차가 안 준다.
                     → **버그다. 고쳐야 한다.**

**"없음" 과 "괜찮음" 이 같은 출력으로 나오면 검사가 아니다.** 그래서 이 계측기는
잡음과 편향을 같은 실행에서 각각 숫자로 찍는다. 한쪽만 재면 구분이 안 된다.

무엇을 찍나
-----------
    single   K개 표본 각각의 GT 오차 (= eval_holdout_error.py 가 재던 값)
    spread   K개 표본이 자기들 평균에서 얼마나 흩어지나  ← 잡음 크기
    mean_K   K개를 평균낸 예측의 GT 오차               ← 편향 크기의 상한 추정
    identity 가만히 있기 기준선

판정
----
    spread ≈ single  이고  mean_K << single   → (a) 표집 잡음 지배
    spread ≈ 0       이고  mean_K ≈ single    → (b) 계통 편향
    둘 다 유의미                               → 섞여 있다. mean_K 가 편향분이다

⚠️ mean_K 는 K=8 에서의 **추정치**다. K→∞ 에서 편향에 수렴한다. 유한 K 에서는
   잡음분 1/sqrt(K) 가 남아 있으므로 편향을 **과대** 추정한다. 하한이 아니라 상한이다.

⚠️ 회전 평균은 rot6d 를 산술평균한 뒤 Gram-Schmidt 로 재직교화한 것이다(현악 L2 평균의
   근사). 측지 평균이 아니므로 큰 각도에서는 정확하지 않다. 여기 값은 5도 규모라 무방하다.

Usage
-----
  python diag_action_noise_floor.py --selftest
  python diag_action_noise_floor.py --checkpoint outputs/e2_A_s0/checkpoints/latest.ckpt \
      --dataset outputs/ds_real_train.zarr.zip --out outputs/noise_A_s0_train.json --k 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_holdout_error import (  # noqa: E402  같은 지표 구현을 공유한다. 두 벌 쓰지 않는다
    GAP_COL,
    POS_COLS,
    ROT_COLS,
    identity_pred,
    metrics,
    rot6d_to_matrix_t,
    selftest as eval_selftest,
    wilson_free_ci,
)

KEYS = ("trans_mm", "rot_deg", "gap_mm")
UNITS = {"trans_mm": "mm", "rot_deg": "도", "gap_mm": "mm"}


def mean_over_k(stack):
    """Arithmetic mean over the K axis; rot6d is re-orthogonalized downstream.
    K축 산술평균. rot6d 는 metrics 안 Gram-Schmidt 가 재직교화한다."""
    return stack.mean(dim=0)


def verdict(single: float, spread: float, mean_k: float) -> str:
    """Classify an error channel as noise-dominated or bias-dominated.
    한 채널의 오차가 잡음 지배인지 편향 지배인지 분류한다."""
    if single < 1e-9:
        return "오차 없음"
    r_spread = spread / single
    r_mean = mean_k / single
    if r_mean < 0.5 and r_spread > 0.5:
        return "표집 잡음 지배 → 버그 아님. K평균/표집스텝으로 줄어든다"
    if r_mean > 0.8 and r_spread < 0.3:
        return "계통 편향 지배 → 규약·정규화·정렬을 의심하라"
    return f"혼재 (편향분 약 {mean_k:.3f}, 잡음분 약 {spread:.3f})"


# ── 자체 검증 — 정답을 아는 행만 쓴다 ───────────────────────────────────────

def _base_gt(n: int = 6, h: int = 8):
    import torch
    g = torch.zeros(n, h, 10)
    g[..., 3] = 1.0
    g[..., 7] = 1.0
    g[..., POS_COLS] = torch.randn(n, h, 3) * 0.005
    g[..., GAP_COL] = 0.04
    return g


def _channels(stack, gt):
    """single / spread / mean_K for one stack of K predictions.
    K개 예측 묶음에 대한 single·spread·mean_K."""
    import torch
    k = stack.shape[0]
    mk = mean_over_k(stack)
    single = {key: 0.0 for key in KEYS}
    spread = {key: 0.0 for key in KEYS}
    for i in range(k):
        ms = metrics(stack[i], gt)
        mp = metrics(stack[i], mk)
        for key in KEYS:
            single[key] += float(ms[key].mean()) / k
            spread[key] += float(mp[key].mean()) / k
    mm = metrics(mk, gt)
    return single, spread, {key: float(mm[key].mean()) for key in KEYS}


def selftest() -> int:
    import torch
    torch.manual_seed(0)
    bad = 0
    K = 8

    print("— 공유 지표 구현(eval_holdout_error) 자체검증 —")
    bad += eval_selftest()
    print("\n— 잡음/편향 분리 자체검증 —")

    # [1] 순수 계통 편향 10mm: 표본이 전부 같다 → spread 0 · mean_K 10mm
    gt = _base_gt()
    bias = gt.clone()
    bias[..., 0] += 0.010
    st = torch.stack([bias] * K)
    s, sp, mk = _channels(st, gt)
    ok = abs(s["trans_mm"] - 10) < 0.05 and sp["trans_mm"] < 1e-3 and abs(mk["trans_mm"] - 10) < 0.05
    print(f"[1] 순수 편향 10mm  single {s['trans_mm']:.3f} spread {sp['trans_mm']:.3f} "
          f"mean_K {mk['trans_mm']:.3f}  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)
    v = verdict(s["trans_mm"], sp["trans_mm"], mk["trans_mm"])
    ok2 = v.startswith("계통 편향")
    print(f"    판정: {v}  ", end=""); print("OK" if ok2 else "!! 실패"); bad += (not ok2)

    # [2] 순수 표집 잡음: 평균 0 · 표준편차 10mm → mean_K 는 약 1/sqrt(K) 로 줄어야 한다
    noise = torch.stack([gt.clone() for _ in range(K)])
    noise[..., 0] += torch.randn(K, *gt.shape[:-1]) * 0.010
    s, sp, mk = _channels(noise, gt)
    ok = sp["trans_mm"] > 0.5 * s["trans_mm"] and mk["trans_mm"] < 0.6 * s["trans_mm"]
    print(f"[2] 순수 잡음 sd10mm single {s['trans_mm']:.3f} spread {sp['trans_mm']:.3f} "
          f"mean_K {mk['trans_mm']:.3f}  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)
    v = verdict(s["trans_mm"], sp["trans_mm"], mk["trans_mm"])
    ok2 = v.startswith("표집 잡음")
    print(f"    판정: {v}  ", end=""); print("OK" if ok2 else "!! 실패"); bad += (not ok2)

    # [3] 회전 잡음 5도: 무작위 축으로 ±5도 → spread 는 5도 규모, mean_K 는 작아야 한다
    ax = torch.randn(K, *gt.shape[:-1], 3)
    ax = ax / ax.norm(dim=-1, keepdim=True)
    ang = torch.full(ax.shape[:-1], np.deg2rad(5.0))
    kx = torch.zeros(*ax.shape[:-1], 3, 3)
    kx[..., 0, 1] = -ax[..., 2]; kx[..., 0, 2] = ax[..., 1]
    kx[..., 1, 0] = ax[..., 2];  kx[..., 1, 2] = -ax[..., 0]
    kx[..., 2, 0] = -ax[..., 1]; kx[..., 2, 1] = ax[..., 0]
    eye = torch.eye(3).expand_as(kx)
    a = ang[..., None, None]
    R = eye + torch.sin(a) * kx + (1 - torch.cos(a)) * (kx @ kx)     # 로드리게스
    rot = torch.stack([gt.clone() for _ in range(K)])
    rot[..., 3:6] = R[..., 0, :]
    rot[..., 6:9] = R[..., 1, :]
    s, sp, mk = _channels(rot, gt)
    ok = abs(s["rot_deg"] - 5.0) < 0.2 and mk["rot_deg"] < 0.6 * s["rot_deg"]
    print(f"[3] 회전 잡음 5도    single {s['rot_deg']:.3f} spread {sp['rot_deg']:.3f} "
          f"mean_K {mk['rot_deg']:.3f}  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    # [4] 편향+잡음 혼합: mean_K 가 편향분(10mm)을 되찾아야 한다
    mix = torch.stack([gt.clone() for _ in range(K)])
    mix[..., 0] += 0.010 + torch.randn(K, *gt.shape[:-1]) * 0.010
    s, sp, mk = _channels(mix, gt)
    ok = 8.0 < mk["trans_mm"] < 14.0
    print(f"[4] 편향10+잡음10    single {s['trans_mm']:.3f} spread {sp['trans_mm']:.3f} "
          f"mean_K {mk['trans_mm']:.3f} (편향 10mm 회수 기대)  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


# ── 본체 ─────────────────────────────────────────────────────────────────

def run(a) -> None:
    import dill
    import hydra
    import torch
    from umi_adapter.upstream import config, enable
    enable()

    torch.manual_seed(a.seed)
    payload = torch.load(a.checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    key = "ema_model" if cfg.training.use_ema else "model"
    policy.load_state_dict(payload["state_dicts"][key])
    policy.to(a.device).eval()
    horizon = int(cfg.task.action_horizon) if "action_horizon" in cfg.task else None
    print(f"체크포인트 {a.checkpoint}")
    print(f"  epoch {dill.loads(payload['pickles']['epoch'])} · action_horizon {horizon} · K={a.k}")
    if horizon is not None and horizon != a.horizon:
        raise SystemExit(f"!! 체크포인트 horizon {horizon} 과 평가 horizon {a.horizon} 이 다르다")

    c = config([f"task.dataset_path={Path(a.dataset).resolve().as_posix()}",
                f"task.action_horizon={a.horizon}",
                f"task.obs_down_sample_steps={a.down_sample}",
                "task.dataset.action_padding=true",
                "task.dataset.val_ratio=0.0"])
    ds = hydra.utils.instantiate(c.task.dataset)
    n_total = len(ds)
    limit = n_total if a.limit <= 0 else min(a.limit, n_total)
    print(f"데이터셋 표본 {limit} / 전체 {n_total}")          # 모수를 같이 찍는다

    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0)
    acc = {name: {key: [] for key in KEYS} for name in ("single", "spread", "mean_k", "identity")}
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            if n_seen >= limit:
                break
            obs = {k2: v.to(a.device) for k2, v in batch["obs"].items()}
            gt = batch["action"].to(a.device)
            preds = []
            for _ in range(a.k):
                out = policy.predict_action(obs)
                p = out["action_pred"] if "action_pred" in out else out["action"]
                preds.append(p[:, :gt.shape[1]])
            st = torch.stack(preds, dim=0)                      # (K, B, H, 10)
            mk = mean_over_k(st)
            for i in range(a.k):
                ms = metrics(st[i], gt)
                mp = metrics(st[i], mk)
                for key in KEYS:
                    acc["single"][key].append(ms[key].mean(dim=-1).cpu().numpy() / a.k)
                    acc["spread"][key].append(mp[key].mean(dim=-1).cpu().numpy() / a.k)
            mm = metrics(mk, gt)
            cur_gap = obs["robot0_gripper_width"][:, -1, 0:1].expand(-1, gt.shape[1])
            mi = metrics(identity_pred(gt, cur_gap), gt)
            for key in KEYS:
                acc["mean_k"][key].append(mm[key].mean(dim=-1).cpu().numpy())
                acc["identity"][key].append(mi[key].mean(dim=-1).cpu().numpy())
            n_seen += gt.shape[0]

    if n_seen == 0:
        raise SystemExit("!! 표본 0. 데이터셋 경로와 down_sample 을 확인하라")

    res = {}
    for name in ("mean_k", "identity"):
        res[name] = {key: round(float(np.concatenate(acc[name][key]).mean()), 4) for key in KEYS}
    for name in ("single", "spread"):
        # K개 조각을 이미 1/K 로 나눠 담았으므로 K개를 더해야 한 표본의 평균이 된다
        res[name] = {}
        for key in KEYS:
            arr = np.concatenate(acc[name][key])
            res[name][key] = round(float(arr.sum() / (len(arr) / a.k)), 4)

    rep = {"checkpoint": str(Path(a.checkpoint).resolve()),
           "dataset": str(Path(a.dataset).resolve()),
           "k": a.k, "samples": int(n_seen), "samples_total": int(n_total),
           "horizon": a.horizon, "down_sample": a.down_sample, "seed": a.seed,
           "note": "single=표본별 GT오차 · spread=표본들의 K평균 대비 흩어짐(잡음) · "
                   "mean_K=K평균 예측의 GT오차(편향 상한 추정) · identity=가만히 있기",
           **res, "verdict": {}}
    for key in KEYS:
        rep["verdict"][key] = verdict(res["single"][key], res["spread"][key], res["mean_k"][key])

    Path(a.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"{'채널':10s} {'single':>9s} {'spread':>9s} {'mean_K':>9s} {'identity':>9s}")
    for key in KEYS:
        print(f"{key:10s} {res['single'][key]:9.3f} {res['spread'][key]:9.3f} "
              f"{res['mean_k'][key]:9.3f} {res['identity'][key]:9.3f}  {UNITS[key]}")
    print("-" * 74)
    for key in KEYS:
        base = res["identity"][key]
        imp = 100 * (base - res["mean_k"][key]) / base if base > 1e-9 else float("nan")
        print(f"{key:10s} {rep['verdict'][key]}")
        print(f"{'':10s}   K평균이 identity 대비 {imp:+.1f}%")
    print(f"→ {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--checkpoint"); ap.add_argument("--dataset"); ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=8, help="같은 관측을 몇 번 표집하나")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--down-sample", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=240, help="표본 상한. 0이면 전부")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    for need in ("checkpoint", "dataset", "out"):
        if not getattr(a, need):
            raise SystemExit(f"!! --{need} 가 필요하다")
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 진단하지 않는다")
    print()
    run(a)


if __name__ == "__main__":
    main()
