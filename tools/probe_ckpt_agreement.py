#!/usr/bin/env python3
"""Are two checkpoints distinguishable at all, given the sampler's own variation?
두 체크포인트가 **샘플러 자체의 변동보다 크게** 다른가.

왜 필요한가
-----------
같은 데이터·같은 설정으로 학습한 두 ckpt 가 있다. 개루프 배수는 3.52x 와 3.67x —
4% 차이다. "실물에서 행동이 다를까?" 라는 질문에 그 숫자는 답하지 못한다.

**diffusion policy 는 확률적 샘플러다.** 같은 ckpt, 같은 관측이어도 추론할 때마다
청크가 다르다. 그 내부 변동이 두 ckpt 의 차이보다 크면 **실물에서 구분 자체가 안 된다.**

그래서 재는 것은 하나다:

    between / within  =  (두 ckpt 사이 거리) / (한 ckpt 안의 반복 간 거리)

    < 1.2   두 ckpt 를 구분할 수 없다. 어느 쪽을 쓰든 같은 분포에서 뽑는 것과 같다
    >= 1.2  구분될 수 있다. **어느 쪽이 좋은지는 이 도구가 말하지 않는다** — 별개 문제다

게이트 1.2 는 **결과 보기 전에 정한 값**이다 (2026-09-21). 빗나가면 게이트를 옮기지
말고 반복 수를 올린다.

이 계측기가 스스로를 못 믿는 지점
----------------------------------
1. **within 이 0 이면** 샘플러가 결정적이라는 뜻이다. 그러면 비율이 무한이 되어
   "아주 다르다"가 **공짜로** 나온다. 그 경우 비율을 내지 않고 미판정으로 둔다.
2. **between 도 within 도 0 이면** 두 ckpt 가 같은 파일일 가능성이 크다. 미판정.
3. 관측 한 벌로 판정하지 않는다. 여러 벌을 돌리고 **모수를 같이 찍는다.**

⚠️ 합성 관측(회색 이미지 + 0 저차원)이다. **실제 카메라 프레임이 아니다.**
   실물 장면에서의 차이는 이것으로 말할 수 없다. 여기서 "구분 불가"가 나와도
   그것은 *이 관측에서* 구분 불가라는 뜻이다.

Usage
    python probe_ckpt_agreement.py --selftest
    python probe_ckpt_agreement.py --ckpt A.ckpt --ckpt B.ckpt --obs 4 --repeat 6 --out out/agree.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

GATE_RATIO = 1.2          # 사전 등록. 결과 보고 옮기지 않는다
MIN_WITHIN_MM = 1e-6      # 이보다 작으면 "변동 없음"으로 보고 비율을 내지 않는다


def rot6d_to_matrix(row6: np.ndarray) -> np.ndarray:
    """rot6d (두 **행**) -> 회전행렬. 열이 아니라 행이다 (v10 규약)."""
    a, b = np.asarray(row6[:3], float), np.asarray(row6[3:6], float)
    r0 = a / (np.linalg.norm(a) + 1e-12)
    b = b - np.dot(r0, b) * r0
    r1 = b / (np.linalg.norm(b) + 1e-12)
    return np.stack([r0, r1, np.cross(r0, r1)], axis=0)


def chunk_distance(a: np.ndarray, b: np.ndarray) -> dict:
    """Distance between two action chunks. 두 액션 청크 사이 거리. 단위 mm / deg."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] < 10:
        raise ValueError(f"청크 모양 {a.shape} vs {b.shape} — (H,10) 이어야 한다")
    pos = np.linalg.norm(a[:, :3] - b[:, :3], axis=1) * 1000.0        # 스텝마다 mm
    gap = np.abs(a[:, 9] - b[:, 9]) * 1000.0
    deg = []
    for ra, rb in zip(a[:, 3:9], b[:, 3:9]):
        R = rot6d_to_matrix(ra) @ rot6d_to_matrix(rb).T
        c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        deg.append(np.degrees(np.arccos(c)))
    return {"pos_mm": float(pos.mean()), "pos_max_mm": float(pos.max()),
            "gap_mm": float(gap.mean()), "rot_deg": float(np.mean(deg))}


def _mean(ds: list[dict], key: str) -> float:
    return float(np.mean([d[key] for d in ds])) if ds else float("nan")


def compare(chunks: dict[str, list[list[np.ndarray]]]) -> dict:
    """chunks[name][obs_index] = [반복별 청크]. 모수를 전부 같이 낸다."""
    names = sorted(chunks)
    if len(names) < 2:
        raise ValueError(f"ckpt 가 {len(names)}개 — 둘 이상이어야 비교가 된다")
    n_obs = len(chunks[names[0]])
    within, between, n_w, n_b = [], [], 0, 0
    for i in range(n_obs):
        for nm in names:                                  # 같은 ckpt 안의 반복 쌍
            reps = chunks[nm][i]
            for x, y in itertools.combinations(range(len(reps)), 2):
                within.append(chunk_distance(reps[x], reps[y])); n_w += 1
        for a, b in itertools.combinations(names, 2):      # 서로 다른 ckpt 쌍
            for x in chunks[a][i]:
                for y in chunks[b][i]:
                    between.append(chunk_distance(x, y)); n_b += 1

    w, bt = _mean(within, "pos_mm"), _mean(between, "pos_mm")
    if not np.isfinite(w) or w < MIN_WITHIN_MM:
        ratio, verdict, why = None, "INCOMPLETE", (
            f"샘플러 변동이 {w:.6f}mm 다 — 0 이면 비율이 무한이 되어 '아주 다르다'가 "
            "공짜로 나온다. 비율을 내지 않는다")
    else:
        ratio = bt / w
        verdict = "구분 가능" if ratio >= GATE_RATIO else "구분 불가"
        why = f"between {bt:.3f}mm / within {w:.3f}mm = {ratio:.2f} (게이트 {GATE_RATIO})"
    return {
        "ckpts": names, "n_obs": n_obs,
        "pairs": {"within": n_w, "between": n_b},
        "within_mm": w, "between_mm": bt, "ratio": ratio,
        "within_rot_deg": _mean(within, "rot_deg"), "between_rot_deg": _mean(between, "rot_deg"),
        "within_gap_mm": _mean(within, "gap_mm"), "between_gap_mm": _mean(between, "gap_mm"),
        "gate_ratio": GATE_RATIO, "verdict": verdict, "why": why,
        "note": "합성 관측이다. 실제 카메라 프레임에서의 차이는 말하지 않는다",
    }


# ── 실제 추론 ────────────────────────────────────────────────────────────────
def load_policy(ckpt: Path, umi_root: str | None):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from smoke_deploy_ckpt import _prepare_umi_path
    print(f"  {_prepare_umi_path(umi_root)}")
    import hydra
    p = torch.load(ckpt, map_location="cpu", weights_only=False)
    policy = hydra.utils.instantiate(p["cfg"].policy)
    policy.load_state_dict(p["state_dicts"]["ema_model"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return policy.to(dev).eval(), p["cfg"], dev


def make_obs(cfg, dev, seed: int):
    """Deterministic synthetic observation. 시드로 고정된 합성 관측."""
    import torch
    g = torch.Generator().manual_seed(seed)
    batch = {}
    for k in cfg.shape_meta.obs:
        shp = list(cfg.shape_meta.obs[k]["shape"])
        h = int(cfg.shape_meta.obs[k]["horizon"])
        if cfg.shape_meta.obs[k].get("type") == "rgb":
            t = 0.5 + 0.05 * torch.randn([1, h] + shp, generator=g)
            t = t.clamp(0.0, 1.0)
        else:
            t = 0.01 * torch.randn([1, h] + shp, generator=g)
        batch[k] = t.to(dev)
    return batch


def run(ckpts: list[Path], n_obs: int, repeat: int, umi_root: str | None) -> dict:
    import torch
    out: dict[str, list[list[np.ndarray]]] = {}
    ref_cfg = None
    for c in ckpts:
        policy, cfg, dev = load_policy(c, umi_root)
        ref_cfg = ref_cfg or cfg
        per_obs = []
        for i in range(n_obs):
            batch = make_obs(ref_cfg, dev, seed=1000 + i)   # 모든 ckpt 에 **같은** 관측
            reps = []
            for r in range(repeat):
                torch.manual_seed(7000 + r)                  # 반복마다 다른 샘플러 시드
                with torch.inference_mode():
                    a = policy.predict_action(batch)["action"][0]
                reps.append(a.detach().cpu().numpy())
            per_obs.append(reps)
            print(f"  {c.name:<34} obs {i + 1}/{n_obs} · 반복 {repeat}", flush=True)
        out[c.name] = per_obs
        del policy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return compare(out)


# ── 자체검증 (torch 없이 돈다) ───────────────────────────────────────────────
def _chunks(rng, n_obs=2, repeat=3, spread=1.0, offset_mm=0.0):
    out = []
    for _ in range(n_obs):
        reps = []
        for _ in range(repeat):
            c = np.zeros((16, 10))
            c[:, :3] = rng.normal(scale=spread * 1e-3, size=(16, 3)) + offset_mm * 1e-3
            c[:, 3:9] = np.tile([1, 0, 0, 0, 1, 0], (16, 1))
            c[:, 9] = 0.045
            reps.append(c)
        out.append(reps)
    return out


def selftest() -> int:
    ok = tot = 0

    def chk(name, cond, note=""):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(f"  {'OK  ' if cond else '실패'} {name}   {note}")

    # [1] 같은 분포 -> 비율 약 1 (구분 불가)
    r = compare({"A": _chunks(np.random.default_rng(1), spread=1.0),
                 "B": _chunks(np.random.default_rng(2), spread=1.0)})
    chk("1 같은 분포 -> 구분 불가", r["verdict"] == "구분 불가" and 0.8 < r["ratio"] < 1.2,
        f"비율 {r['ratio']:.2f} · within {r['within_mm']:.2f}mm")

    # [2] B 에 10mm 오프셋 -> 구분 가능 (정답 아는 행)
    r2 = compare({"A": _chunks(np.random.default_rng(1), spread=1.0),
                  "B": _chunks(np.random.default_rng(2), spread=1.0, offset_mm=10.0)})
    chk("2 10mm 오프셋 -> 구분 가능 (정답 아는 행)",
        r2["verdict"] == "구분 가능" and r2["between_mm"] > 9.0,
        f"비율 {r2['ratio']:.2f} · between {r2['between_mm']:.2f}mm")

    # [3] within 0 -> 미판정. '아주 다르다'가 공짜로 나오면 안 된다 (판별행)
    fixed = [[np.tile(np.array([[0.0] * 3 + [1, 0, 0, 0, 1, 0] + [0.045]]), (16, 1))] * 3]
    moved = [[np.tile(np.array([[0.01] * 3 + [1, 0, 0, 0, 1, 0] + [0.045]]), (16, 1))] * 3]
    r3 = compare({"A": fixed, "B": moved})
    chk("3 샘플러 변동 0 -> 미판정 (판별행)",
        r3["verdict"] == "INCOMPLETE" and r3["ratio"] is None, r3["why"][:40])

    # [4] 완전히 같은 ckpt 두 벌 -> between 0, within 0 -> 미판정
    r4 = compare({"A": fixed, "B": [list(fixed[0])]})
    chk("4 동일 파일 -> 미판정 (판별행)", r4["verdict"] == "INCOMPLETE",
        f"between {r4['between_mm']:.6f}mm")

    # [5] 단위: 10mm 오프셋이 10mm 로 나온다
    d = chunk_distance(np.hstack([np.zeros((16, 3)), np.tile([1, 0, 0, 0, 1, 0], (16, 1)),
                                  np.full((16, 1), 0.045)]),
                       np.hstack([np.full((16, 3), 0.0), np.tile([1, 0, 0, 0, 1, 0], (16, 1)),
                                  np.full((16, 1), 0.045)]) + np.hstack(
                           [np.tile([0.01, 0, 0], (16, 1)), np.zeros((16, 7))]))
    chk("5 단위 mm (정답 아는 행)", abs(d["pos_mm"] - 10.0) < 1e-6, f"{d['pos_mm']:.4f} mm")

    # [6] 회전 90도가 90도로 나온다
    a = np.hstack([np.zeros((16, 3)), np.tile([1, 0, 0, 0, 1, 0], (16, 1)), np.full((16, 1), 0.045)])
    b = np.hstack([np.zeros((16, 3)), np.tile([0, -1, 0, 1, 0, 0], (16, 1)), np.full((16, 1), 0.045)])
    chk("6 회전 90도 (정답 아는 행)", abs(chunk_distance(a, b)["rot_deg"] - 90.0) < 1e-6,
        f"{chunk_distance(a, b)['rot_deg']:.3f} deg")

    # [7] 순서 무관
    chk("7 A·B 순서 무관", abs(compare({"B": _chunks(np.random.default_rng(1)),
                                       "A": _chunks(np.random.default_rng(2))})["ratio"]
                              - compare({"A": _chunks(np.random.default_rng(2)),
                                         "B": _chunks(np.random.default_rng(1))})["ratio"]) < 1e-9)

    # [8] 모수 보고
    chk("8 모수 보고", r["pairs"]["within"] == 2 * 2 * 3 and r["pairs"]["between"] == 2 * 9,
        f"within {r['pairs']['within']} · between {r['pairs']['between']} 쌍 · obs {r['n_obs']}")

    # [9] ckpt 하나면 거부
    try:
        compare({"A": _chunks(np.random.default_rng(1))}); bad = False
    except ValueError:
        bad = True
    chk("9 ckpt 1개 -> 거부 (판별행)", bad)

    print(f"\n자체검증 {ok}/{tot}")
    return 0 if ok == tot else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", action="append", default=[], help="두 번 이상 준다")
    ap.add_argument("--obs", type=int, default=4, help="합성 관측 벌 수")
    ap.add_argument("--repeat", type=int, default=6, help="관측 한 벌당 반복 추론 수")
    ap.add_argument("--umi-root")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if len(a.ckpt) < 2:
        ap.error("--ckpt 를 둘 이상 줘라 (또는 --selftest)")

    print("계측기 자체검증 먼저 —")
    if selftest() != 0:
        raise SystemExit("!! 자체검증 실패 — 실데이터를 재지 않는다")
    print()
    r = run([Path(c).expanduser() for c in a.ckpt], a.obs, a.repeat, a.umi_root)

    print(f"\n비교 {r['ckpts']}")
    print(f"관측 {r['n_obs']}벌 · within {r['pairs']['within']}쌍 · between {r['pairs']['between']}쌍")
    print(f"위치   within {r['within_mm']:8.3f} mm   between {r['between_mm']:8.3f} mm")
    print(f"회전   within {r['within_rot_deg']:8.3f} deg  between {r['between_rot_deg']:8.3f} deg")
    print(f"개구   within {r['within_gap_mm']:8.3f} mm   between {r['between_gap_mm']:8.3f} mm")
    print(f"\n판정   {r['verdict']}   {r['why']}")
    print(f"** {r['note']} **")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
