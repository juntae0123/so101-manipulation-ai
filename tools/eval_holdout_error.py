"""Holdout action-chunk prediction error for E2 (real UMI demos).
E2 판정기 — 홀드아웃 실 시연의 상대 action chunk 예측 오차를 잰다.

무엇을 재나 (D-AI-52)
---------------------
실물 로봇팔이 고장나 롤아웃 평가가 불가능하다. 그래서 **학습에 쓰지 않은
실 시연 홀드아웃**에서 정책이 다음 0.8초 궤적을 얼마나 맞히는지를 본다.

⚠️ **이것은 롤아웃이 아니다.** val_loss 0.00547→0.00517 인데 롤아웃 0% 였던
   전례가 있다. 그래서 **identity(가만히 있기) 대비 개선율**을 반드시 병기한다.
   결론은 "홀드아웃 실 시연 예측 오차를 낮췄다" 까지만 쓴다.

지표 (트랙 A 명세)
- translation L2 [mm] · rotation geodesic [deg] · gap MAE [mm]
- horizon 1~8 별 + 8-step chunk 전체
- identity 기준 대비 개선율
- 주 비교는 **같은 홀드아웃 에피소드에 대한 paired 차이** + 에피소드 단위 신뢰구간

Usage
-----
  python eval_holdout_error.py --selftest
  python eval_holdout_error.py --checkpoint outputs/e2_A_s0/checkpoints/latest.ckpt \
      --dataset outputs/ds_real_holdout.zarr.zip --out outputs/err_A_s0.json
  python eval_holdout_error.py --compare outputs/err_B_s0.json outputs/err_A_s0.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROT_COLS = slice(3, 9)
POS_COLS = slice(0, 3)
GAP_COL = 9


def rot6d_to_matrix_t(v):
    """Batch 6D (first two ROWS) -> rotation matrices. 6D(행 규약) → 회전행렬."""
    import torch
    a0, a1 = v[..., 0:3], v[..., 3:6]
    b0 = a0 / (a0.norm(dim=-1, keepdim=True) + 1e-12)
    b1 = a1 - (b0 * a1).sum(-1, keepdim=True) * b0
    b1 = b1 / (b1.norm(dim=-1, keepdim=True) + 1e-12)
    b2 = torch.cross(b0, b1, dim=-1)
    return torch.stack([b0, b1, b2], dim=-2)          # 행 기준


def geodesic_deg_t(Ra, Rb):
    """Batch geodesic angle [deg]. 배치 측지 각도[도]."""
    import torch
    m = torch.matmul(Ra.transpose(-1, -2), Rb)
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    c = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.acos(c))


def metrics(pred, gt):
    """Per-sample, per-horizon errors in physical units. 표본·지평별 물리 단위 오차."""
    import torch
    d = {}
    d["trans_mm"] = (pred[..., POS_COLS] - gt[..., POS_COLS]).norm(dim=-1) * 1000.0
    d["rot_deg"] = geodesic_deg_t(rot6d_to_matrix_t(pred[..., ROT_COLS]),
                                  rot6d_to_matrix_t(gt[..., ROT_COLS]))
    d["gap_mm"] = (pred[..., GAP_COL] - gt[..., GAP_COL]).abs() * 1000.0
    return d


def identity_pred(gt, cur_gap):
    """The 'stay still' predictor: zero translation, identity rotation, hold gap.
    '가만히 있기' 예측기 — 이동 0, 회전 항등, gap 유지. 이걸 못 이기면 학습이 아니다."""
    import torch
    out = torch.zeros_like(gt)
    out[..., 3] = 1.0; out[..., 7] = 1.0          # r0=(1,0,0) r1=(0,1,0)
    out[..., GAP_COL] = cur_gap
    return out


def wilson_free_ci(vals):
    """Mean with a normal CI over episodes. 에피소드 단위 평균과 신뢰구간."""
    v = np.asarray(vals, dtype=np.float64)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "ci": None}
    m = float(v.mean())
    if n == 1:
        return {"n": 1, "mean": m, "ci": None}
    se = float(v.std(ddof=1) / math.sqrt(n))
    return {"n": n, "mean": m, "ci": [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)],
            "sd": round(float(v.std(ddof=1)), 4)}


# ── 자체 검증 ────────────────────────────────────────────────────────────

def selftest() -> int:
    import torch
    bad = 0
    # 1) 완전 일치면 오차 0
    g = torch.zeros(4, 8, 10); g[..., 3] = 1.0; g[..., 7] = 1.0
    g[..., POS_COLS] = torch.randn(4, 8, 3) * 0.01
    g[..., GAP_COL] = 0.04
    m = metrics(g.clone(), g)
    e = max(float(m[k].abs().max()) for k in m)
    print(f"[1] 동일 입력 오차 {e:.3e}", end="  ")
    print("OK" if e < 1e-4 else "!! 실패"); bad += e >= 1e-4
    # 2) 10mm 어긋내면 10mm 로 나오나
    p = g.clone(); p[..., 0] += 0.01
    m = metrics(p, g)
    e = abs(float(m["trans_mm"].mean()) - 10.0)
    print(f"[2] 10mm 오프셋 → {float(m['trans_mm'].mean()):.3f}mm", end="  ")
    print("OK" if e < 1e-3 else "!! 실패"); bad += e >= 1e-3
    # 3) 90도 회전이 90도로 나오나
    p = g.clone(); p[..., 3:9] = torch.tensor([0., 1., 0., -1., 0., 0.])
    m = metrics(p, g)
    e = abs(float(m["rot_deg"].mean()) - 90.0)
    print(f"[3] 90도 회전 → {float(m['rot_deg'].mean()):.3f}도", end="  ")
    print("OK" if e < 1e-2 else "!! 실패"); bad += e >= 1e-2
    # 4) identity 예측기가 '움직임 없음'을 정확히 맞히나
    still = torch.zeros(2, 8, 10); still[..., 3] = 1.0; still[..., 7] = 1.0
    still[..., GAP_COL] = 0.05
    idp = identity_pred(still, torch.full((2, 8), 0.05))
    m = metrics(idp, still)
    e = max(float(m[k].abs().max()) for k in m)
    print(f"[4] identity 예측기 정지 궤적 오차 {e:.3e}", end="  ")
    print("OK" if e < 1e-4 else "!! 실패"); bad += e >= 1e-4
    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


# ── 본체 ─────────────────────────────────────────────────────────────────

def run(a) -> None:
    import torch, dill, hydra
    from umi_adapter.upstream import enable, config
    enable()

    payload = torch.load(a.checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    key = "ema_model" if cfg.training.use_ema else "model"
    policy.load_state_dict(payload["state_dicts"][key])
    policy.to(a.device).eval()
    horizon = int(cfg.task.action_horizon) if "action_horizon" in cfg.task else None
    print(f"체크포인트 {a.checkpoint}")
    print(f"  epoch {dill.loads(payload['pickles']['epoch'])} · action_horizon {horizon}")

    c = config([f"task.dataset_path={Path(a.dataset).resolve().as_posix()}",
                f"task.action_horizon={a.horizon}",
                f"task.obs_down_sample_steps={a.down_sample}",
                "task.dataset.action_padding=true",
                "task.dataset.val_ratio=0.0"])
    ds = hydra.utils.instantiate(c.task.dataset)
    print(f"홀드아웃 데이터셋 샘플 {len(ds)}")
    if horizon is not None and horizon != a.horizon:
        raise SystemExit(f"!! 체크포인트 horizon {horizon} 과 평가 horizon {a.horizon} 이 다르다")

    # 표본 → 에피소드 매핑. 없으면 pooled 만 내고 그 사실을 밝힌다
    ep_of = None
    try:
        idx = np.asarray(ds.sampler.indices)
        ends = np.asarray(ds.replay_buffer.episode_ends)
        ep_of = np.searchsorted(ends, idx[:, 0], side="right")
        print(f"  에피소드 매핑 OK — {len(set(ep_of.tolist()))}개 에피소드")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  !! 에피소드 매핑 실패 ({type(exc).__name__}). "
              f"에피소드 단위 신뢰구간을 내지 않는다")

    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0)
    per = {k: [] for k in ("trans_mm", "rot_deg", "gap_mm")}
    idn = {k: [] for k in per}
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            obs = {k: v.to(a.device) for k, v in batch["obs"].items()}
            gt = batch["action"].to(a.device)
            out = policy.predict_action(obs)
            pred = out["action_pred"] if "action_pred" in out else out["action"]
            pred = pred[:, :gt.shape[1]]
            m = metrics(pred, gt)
            cur_gap = obs["robot0_gripper_width"][:, -1, 0:1].expand(-1, gt.shape[1])
            mi = metrics(identity_pred(gt, cur_gap), gt)
            for k in per:
                per[k].append(m[k].cpu().numpy())
                idn[k].append(mi[k].cpu().numpy())
            n_seen += gt.shape[0]
    per = {k: np.concatenate(v, axis=0) for k, v in per.items()}   # (N, H)
    idn = {k: np.concatenate(v, axis=0) for k, v in idn.items()}
    print(f"  평가 표본 {n_seen}")

    rep = {"checkpoint": str(Path(a.checkpoint).resolve()),
           "dataset": str(Path(a.dataset).resolve()),
           "samples": int(n_seen), "horizon": a.horizon,
           "metric_note": "홀드아웃 상대 action chunk 예측 오차. 롤아웃이 아니다. "
                          "결론은 '실 시연 예측 오차를 낮췄다' 까지만.",
           "per_horizon": {}, "chunk": {}, "identity": {}, "improvement_pct": {},
           "per_episode": None}
    for k in per:
        rep["per_horizon"][k] = [round(float(per[k][:, h].mean()), 4)
                                 for h in range(per[k].shape[1])]
        rep["chunk"][k] = round(float(per[k].mean()), 4)
        rep["identity"][k] = round(float(idn[k].mean()), 4)
        base = idn[k].mean()
        rep["improvement_pct"][k] = (round(float(100 * (base - per[k].mean()) / base), 2)
                                     if base > 1e-9 else None)
    if ep_of is not None and len(ep_of) == per["trans_mm"].shape[0]:
        pe = {}
        for k in per:
            pe[k] = {int(e): round(float(per[k][ep_of == e].mean()), 4)
                     for e in sorted(set(ep_of.tolist()))}
        rep["per_episode"] = pe
        rep["episode_level"] = {k: wilson_free_ci(list(pe[k].values())) for k in per}

    Path(a.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 58)
    for k, unit in (("trans_mm", "mm"), ("rot_deg", "도"), ("gap_mm", "mm")):
        imp = rep["improvement_pct"][k]
        flag = "" if (imp is not None and imp > 0) else "   !! identity 보다 나쁘다"
        print(f"{k:10s} chunk {rep['chunk'][k]:8.3f}{unit}  "
              f"identity {rep['identity'][k]:8.3f}{unit}  개선 {imp}%{flag}")
    print(f"horizon별 trans_mm: {rep['per_horizon']['trans_mm']}")
    if rep["per_episode"] is None:
        print("!! 에피소드 단위 집계 없음 — paired 비교를 못 한다")
    print(f"→ {a.out}")


def compare(fa: str, fb: str) -> None:
    """Paired per-episode difference between two runs. 두 실행의 에피소드 단위 짝지은 차이."""
    A = json.loads(Path(fa).read_text(encoding="utf-8"))
    B = json.loads(Path(fb).read_text(encoding="utf-8"))
    if not A.get("per_episode") or not B.get("per_episode"):
        raise SystemExit("!! 한쪽에 에피소드 단위 집계가 없다. paired 비교 불가")
    print(f"A = {Path(fa).name}\nB = {Path(fb).name}\n")
    for k in ("trans_mm", "rot_deg", "gap_mm"):
        ea, eb = A["per_episode"][k], B["per_episode"][k]
        common = sorted(set(ea) & set(eb))
        if not common:
            print(f"{k}: 공통 에피소드 없음"); continue
        d = [eb[e] - ea[e] for e in common]        # B − A. 음수면 B 가 낫다
        s = wilson_free_ci(d)
        sign = "B 가 낫다" if s["mean"] < 0 else "A 가 낫다"
        crosses = s["ci"] is not None and s["ci"][0] < 0 < s["ci"][1]
        verdict = "차이 없음 (구간이 0을 포함)" if crosses else sign
        print(f"{k:10s} B−A 평균 {s['mean']:+.4f}  95%CI {s['ci']}  n={s['n']}  → {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A_json", "B_json"))
    ap.add_argument("--checkpoint"); ap.add_argument("--dataset"); ap.add_argument("--out")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--down-sample", type=int, default=1,
                    help="실데이터 zarr 은 이미 10Hz 라 1. 시뮬 zarr 이면 3")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if a.compare:
        compare(a.compare[0], a.compare[1]); return
    for need in ("checkpoint", "dataset", "out"):
        if not getattr(a, need):
            raise SystemExit(f"!! --{need} 가 필요하다")
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 평가하지 않는다")
    print()
    run(a)


if __name__ == "__main__":
    main()
