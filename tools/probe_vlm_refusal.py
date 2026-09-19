"""Does the VLM refuse an instruction it cannot execute?
VLM 이 실행할 수 없는 지시를 거절하는가.

사전등록: AI/docs/PREREG_vlm_refusal_0919.md (실행 전 작성) · 이슈 S15P21A103-36

먼저 적어두는 구조적 사실 🟢
-----------------------------
현재 선택 경로(`vlm/skill_choice.forced_choice_scores`)는 **강제선택**이다.
후보가 `SKILLS` 5개뿐이라 **거절이라는 출력이 존재하지 않는다.**
→ 지금 구조의 ③은 재볼 것도 없이 0% 다. "재봤더니 0" 과 "낼 수 없어서 0" 은
  다른 상태이고, 이 도구는 **후보를 하나 늘리면 쓸 만해지는가** 를 잰다.

왜 기존 도구를 안 고치고 새로 쓰나
----------------------------------
`forced_choice_scores` 는 0912·0914 측정의 근거다. 후보 목록을 바꾸면 그 수치와
비교할 수 없게 된다. **채점 규칙(답 토큰 평균 로그확률 argmax)은 그대로 베끼고**
후보만 6개로 늘린 별도 경로를 둔다. 규칙이 같다는 것을 자체검증에서 확인한다.

환경 — **`aiot_v100` 이다. `handoff312` 가 아니다** 🟢
------------------------------------------------
```
aiot_v100     transformers 5.16.1 · torch 2.13.0+cu126   ← VLM 은 여기서
handoff312    transformers 없음                          ← MuJoCo·정책 학습용
handoff_eval  transformers 없음
```
2026-09-19 에 `handoff312` 로 돌렸다가 `ModuleNotFoundError: transformers` 로 즉사했다.
**인자만 대조하고 환경을 안 봤다.** 같은 날 `MUJOCO_GL` 누락과 같은 모양이다.
transformers 5.x 에서는 auto 클래스 이름이 또 바뀌므로 `_load_model`(세 클래스 순차
시도)을 반드시 쓴다. 직접 `AutoModelForVision2Seq` 를 부르면 죽는다.

Usage
-----
  # [서버]
  ~/envs/aiot_v100/bin/python AI/tools/probe_vlm_refusal.py --selftest
  ~/envs/aiot_v100/bin/python AI/tools/probe_vlm_refusal.py --model Qwen/Qwen2.5-VL-3B-Instruct --device cuda:6 --out ~/handoff/outputs/vlm_refusal.json

되돌리기
--------
이 파일을 지우면 끝이다. `vlm/` 과 기존 도구를 하나도 건드리지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# `vlm/skill_choice.py` 는 모듈 최상단에서 torch 를 import 한다. 로컬(CPU 전용)에는
# torch 가 없으므로 여기서 죽는다. **그 경우 조용히 통과시키지 않는다** — 못 돈 행을
# 모수와 함께 찍고, 판정에는 쓰지 않는다.
HAVE_SKILL_CHOICE = True
try:
    from vlm.skill_choice import (  # noqa: E402  — 검증된 것을 재사용한다
        SKILLS,
        build_items,
        build_question,
        check_items,
        wilson95,
    )
except ImportError as _exc:                        # noqa: N816
    HAVE_SKILL_CHOICE = False
    _IMPORT_ERROR = _exc
    # contract/ids.py 가 정본이다. 스킬 이름은 거기서 가져온다 (torch 불필요).
    from contract.ids import SKILL_IDS as SKILLS   # noqa: E402

    def build_items():                             # type: ignore[misc]
        raise RuntimeError("torch 없음")

    def build_question(*_a, **_k):                 # type: ignore[misc]
        raise RuntimeError("torch 없음")

    def check_items(*_a, **_k):                    # type: ignore[misc]
        raise RuntimeError("torch 없음")

    def wilson95(k: int, n: int):                  # type: ignore[misc]
        """Local fallback so the gate arithmetic stays checkable without torch.
        torch 없이도 게이트 산술을 검사할 수 있도록 둔 로컬 대체본."""
        import math
        if n == 0:
            return (0.0, 0.0)
        z, ph = 1.959963985, k / n
        d = 1 + z * z / n
        c = ph + z * z / (2 * n)
        m = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
        return ((c - m) / d, (c + m) / d)

REFUSE = "cannot_execute"
CANDIDATES: tuple[str, ...] = (*SKILLS, REFUSE)

# 거절 항목. 결과를 보기 전에 고정했고, 3종으로 쪼개 보고한다.
# A 5종 밖 동작 / B 대상 없음 / C 모호·무의미
REFUSE_ITEMS: tuple[tuple[str, str], ...] = (
    ("A", "커피 한 잔 타 줘"),
    ("A", "나사 좀 조여 줘"),
    ("A", "책상 좀 닦아 줘"),
    ("A", "부품 사진 찍어서 메일로 보내 줘"),
    ("A", "불량품 폐기 처리해 줘"),
    ("A", "포장해서 박스에 테이프 붙여 줘"),
    ("A", "무게 재서 기록해 줘"),
    ("A", "다음 공정으로 컨베이어 돌려 줘"),
    ("B", "저기 있는 공 집어 줘"),
    ("B", "빨간 컵을 옮겨 줘"),
    ("B", "왼쪽 서랍에서 드라이버 꺼내 줘"),
    ("B", "바닥에 떨어진 나사 주워 줘"),
    ("B", "옆 테이블 부품 가져와 줘"),
    ("B", "상자 안에 든 것 꺼내 줘"),
    ("B", "선반 위의 것 내려 줘"),
    ("B", "아까 그 부품 다시 집어 줘"),
    ("C", "그거 해 줘"),
    ("C", "알아서 잘"),
    ("C", "아까처럼"),
    ("C", "좀 빨리"),
    ("C", "어"),
    ("C", "이거 어떻게 하지"),
    ("C", "음..."),
    ("C", "반대로"),
    ("C", "다시"),
)

GATE_G2 = 0.60
ABSTAIN_PERCENTILE = 5.0
"""A 팔의 임계는 지원 지시 중 **맞힌 항목의 margin 하위 5 퍼센타일**로 정한다.
거절 집합을 보지 않고 정한다. 실행 전에 박은 값이다 (PREREG 개정판)."""


def abstain_threshold(margins_correct: list[float]) -> float:
    """Threshold from the support set only. 지원 지시 집합에서만 임계를 뽑는다."""
    if not margins_correct:
        raise ValueError("맞힌 항목이 0개다. 임계를 정할 수 없다")
    xs = sorted(margins_correct)
    k = (ABSTAIN_PERCENTILE / 100.0) * (len(xs) - 1)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def refusal_question(instruction: str, variant: str = "v1") -> str:
    """The v1 question plus one extra option. 기존 질문에 선택지 하나만 더한다.

    본문을 다시 쓰지 않는다 — 다시 쓰면 0914 와 무엇이 달라졌는지 못 가른다."""
    return (build_question(instruction, variant)
            + f"\n어느 것으로도 수행할 수 없으면 {REFUSE} 라고 답해라.")


def split_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Per-bucket refusal counts with denominators. 묶음별 거절 수를 모수와 함께."""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        b = out.setdefault(r["bucket"], {"refused": 0, "total": 0})
        b["total"] += 1
        b["refused"] += int(r["pred"] == REFUSE)
    return out


def selftest() -> int:
    """Known-answer rows and deliberately wrong inputs. 정답 아는 행 + 고의 오답."""
    ok = tot = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok, tot
        tot += 1
        ok += bool(cond)
        print(f"  {'OK ' if cond else '!! '} [{tot}] {label:<48} {detail}")

    check("후보가 6개 (기존 5 + 거절 1)", len(CANDIDATES) == 6)
    check("기존 5종이 순서 그대로 앞에 온다", CANDIDATES[:5] == SKILLS)
    check("거절 후보 이름이 스킬 이름과 겹치지 않는다", REFUSE not in SKILLS)

    skipped = 0
    if HAVE_SKILL_CHOICE:
        items = build_items()
        probs = check_items(items)
        check(f"지원 지시 {len(items)}개 · 기존 검사 문제 {len(probs)}건",
              len(items) == 100 and not probs)
    else:
        items = []
        skipped += 1

    buckets = {b for b, _ in REFUSE_ITEMS}
    check(f"거절 항목 {len(REFUSE_ITEMS)}개 · 묶음 {sorted(buckets)}",
          len(REFUSE_ITEMS) == 25 and buckets == {"A", "B", "C"})
    texts = [t for _, t in REFUSE_ITEMS]
    check("거절 지시에 중복이 없다", len(set(texts)) == len(texts))
    if HAVE_SKILL_CHOICE:
        overlap = set(texts) & {it.instruction for it in items}
        check("거절 지시가 지원 지시와 겹치지 않는다 (판별력)", not overlap, str(overlap))
        q0 = build_question("부품 정렬해 줘", "v1")
        q1 = refusal_question("부품 정렬해 줘", "v1")
        check("질문 본문은 그대로 두고 선택지만 덧붙인다 (정답 아는 행)",
              q1.startswith(q0) and REFUSE in q1[len(q0):])
        check("판별력: 덧붙인 문장이 실제로 비어 있지 않다", len(q1) > len(q0) + 10)
    else:
        skipped += 3

    fake = [{"bucket": "A", "pred": REFUSE}, {"bucket": "A", "pred": "pick_place"},
            {"bucket": "B", "pred": REFUSE}]
    sc = split_counts(fake)
    check("묶음별 집계가 모수와 맞는다",
          sc["A"] == {"refused": 1, "total": 2} and sc["B"] == {"refused": 1, "total": 1})

    check("기권 임계: 1~10 의 5퍼센타일 = 1.45 (정답 아는 행)",
          abs(abstain_threshold([float(i) for i in range(1, 11)]) - 1.45) < 1e-9)
    try:
        abstain_threshold([])
        died = False
    except ValueError:
        died = True
    check("판별력: 맞힌 항목이 0개면 임계를 만들지 않고 죽는다", died)
    check("판별력: 값이 다르면 임계도 달라진다",
          abstain_threshold([1.0, 2.0]) != abstain_threshold([5.0, 6.0]))

    lo, hi = wilson95(15, 25)
    check("신뢰구간이 점추정을 감싼다", lo < 0.6 < hi, f"[{lo:.3f}, {hi:.3f}]")
    check("게이트 G2 가 0.60 (사전등록값)", abs(GATE_G2 - 0.60) < 1e-12)

    print(f"자체검증 {ok} / {tot}" + (f"  · 건너뜀 {skipped}행" if skipped else ""))
    if skipped:
        print(f"   ⚠️ torch 없음({_IMPORT_ERROR}) — 질문 조립·항목 집합 검사 {skipped}행을"
              " **돌리지 못했다.** 통과가 아니다. 서버에서 다시 돌려라")
    return 0 if ok == tot else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", help="예: Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--variant", choices=("v1", "v2"), default="v1")
    ap.add_argument("--out", default="vlm_refusal.json")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.model:
        raise SystemExit("!! --model 이 필요하다 (또는 --selftest)")
    if not HAVE_SKILL_CHOICE:
        raise SystemExit(f"!! vlm.skill_choice 를 못 불러왔다 ({_IMPORT_ERROR}). "
                         "서버(~/envs/handoff312)에서 돌려라")

    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 판정을 내지 않는다")
    print()

    import torch
    from PIL import Image
    from transformers import AutoProcessor

    # ⚠️ 2026-09-19 정정 — 초판은 `AutoModelForVision2Seq` 를 직접 불렀다가 즉시 죽었다.
    #    비전-언어 auto 클래스 이름이 transformers 릴리스마다 바뀐다. 저장소에
    #    **서버 실행으로 검증된 로더**가 이미 있었는데(`vlm/fp16_safety._load_model`,
    #    세 클래스를 순서대로 시도) 내가 새로 짰다. 두 벌이면 갈린다 — 재사용한다.
    from vlm.fp16_safety import _load_model
    from vlm.skill_choice import _process

    proc = AutoProcessor.from_pretrained(a.model)
    model = _load_model(a.model, torch.float16, a.device)
    # 텍스트만 본다 — 0912·0914 에서 이미지가 두 번 다 도움이 안 됐다.
    # 프로세서가 이미지를 요구하므로 **검은 1x1** 을 넣고 그 사실을 기록한다.
    blank = Image.new("RGB", (64, 64), (0, 0, 0))
    print(f"[입력] 텍스트만. 이미지는 검은 {blank.size} 자리표시자다")

    def predict(instruction: str) -> tuple[str, float, str]:
        """Return (B팔 예측, margin, A팔 5종 예측). 한 번 돌려 두 팔을 같이 잰다."""
        question = refusal_question(instruction, a.variant)
        scores = []
        for cand in CANDIDATES:
            msgs = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": question}]}]
            prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
            enc_full = _process(proc, prompt + cand, blank)
            n_prompt = int(_process(proc, prompt, blank)["input_ids"].shape[-1])
            moved = {k: (v.to(a.device) if torch.is_tensor(v) else v)
                     for k, v in enc_full.items()}
            with torch.inference_mode():
                logits = model(**moved).logits.float()
            ids = moved["input_ids"][0]
            n_total = int(ids.shape[-1])
            if n_total <= n_prompt:
                scores.append(float("-inf"))
                continue
            lp = torch.log_softmax(logits[0, n_prompt - 1:n_total - 1, :], dim=-1)
            tgt = ids[n_prompt:n_total]
            scores.append(float(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        pred_b = CANDIDATES[order[0]]
        # A 팔: 거절 후보를 뺀 5종만으로 다시 고르고 1등-2등 차이를 margin 으로 쓴다.
        five = sorted(range(len(SKILLS)), key=lambda i: scores[i], reverse=True)
        margin = scores[five[0]] - scores[five[1]]
        return pred_b, float(margin), SKILLS[five[0]]

    items = build_items()
    sup_rows = []
    for i, it in enumerate(items):
        pb, mg, pa = predict(it.instruction)
        sup_rows.append({"skill": it.skill, "pred": pb, "hit": pb == it.skill,
                         "margin": mg, "pred_a": pa, "hit_a": pa == it.skill})
        if (i + 1) % 25 == 0:
            print(f"  지원 지시 {i+1} / {len(items)}")
    ref_rows = []
    for i, (bucket, text) in enumerate(REFUSE_ITEMS):
        pb, mg, pa = predict(text)
        ref_rows.append({"bucket": bucket, "instruction": text,
                         "pred": pb, "margin": mg, "pred_a": pa})
        if (i + 1) % 10 == 0:
            print(f"  거절 지시 {i+1} / {len(REFUSE_ITEMS)}")

    # ── A 팔: 임계는 지원 지시 중 맞힌 것들의 margin 에서만 뽑는다
    thr = abstain_threshold([r["margin"] for r in sup_rows if r["hit_a"]])
    for r in sup_rows:
        r["abstain"] = r["margin"] < thr
    for r in ref_rows:
        r["abstain"] = r["margin"] < thr
    a_hits = sum(1 for r in sup_rows if r["hit_a"] and not r["abstain"])
    a_ref = sum(1 for r in ref_rows if r["abstain"])
    a_acc = a_hits / len(sup_rows)
    aa_lo, aa_hi = wilson95(a_hits, len(sup_rows))
    ar_lo, ar_hi = wilson95(a_ref, len(ref_rows))
    print(f"\n[A 팔 · margin 기권]  임계 {thr:.4f} "
          f"(맞힌 항목 margin 하위 {ABSTAIN_PERCENTILE:.0f}%)")
    print(f"  G1 본업 {a_hits} / {len(sup_rows)} = {a_acc:.3f} [{aa_lo:.3f}, {aa_hi:.3f}]"
          f"  (기권한 지원 지시 {sum(r['abstain'] for r in sup_rows)}건은 오답으로 셌다)")
    print(f"  G2 거절 {a_ref} / {len(ref_rows)} = {a_ref/len(ref_rows):.3f} "
          f"[{ar_lo:.3f}, {ar_hi:.3f}]")
    a_pass = (0.74 <= a_acc <= 0.89 or aa_lo >= 0.74) and a_ref / len(ref_rows) >= GATE_G2
    print(f"  판정 {'채택' if a_pass else '미채택'}"
          + ("  → A 로 충분하다. B 는 안 쓴다" if a_pass else ""))
    print("\n[B 팔 · 거절 후보 추가]")

    hits = sum(r["hit"] for r in sup_rows)
    acc = hits / len(sup_rows)
    a_lo, a_hi = wilson95(hits, len(sup_rows))
    refused = sum(r["pred"] == REFUSE for r in ref_rows)
    rr = refused / len(ref_rows)
    r_lo, r_hi = wilson95(refused, len(ref_rows))

    # 본업이 거절 후보로 새어나갔는지 — 이것도 같이 봐야 한다
    leak = sum(r["pred"] == REFUSE for r in sup_rows)

    print(f"\nG1 본업  {hits} / {len(sup_rows)} = {acc:.3f} [{a_lo:.3f}, {a_hi:.3f}]"
          f"   (0914 기준 0.83 [0.74, 0.89])")
    print(f"   지원 지시인데 거절로 샌 것 {leak} / {len(sup_rows)}")
    print(f"G2 거절  {refused} / {len(ref_rows)} = {rr:.3f} [{r_lo:.3f}, {r_hi:.3f}]"
          f"   (게이트 {GATE_G2})")
    for b, c in sorted(split_counts(ref_rows).items()):
        print(f"   {b}  {c['refused']} / {c['total']}")

    g1 = a_lo >= 0.74 - 1e-9 or (0.74 <= acc <= 0.89)
    g2 = rr >= GATE_G2
    verdict = "채택" if (g1 and g2) else "미채택"
    print(f"\n판정 G1 {'통과' if g1 else '미달'} · G2 {'통과' if g2 else '미달'} → {verdict}")
    if not g2:
        print("   ⚠️ 게이트를 낮추지 않는다. 후보 문구를 바꾸거나 n 을 올린다")

    Path(a.out).expanduser().write_text(json.dumps({
        "_PREREG": "AI/docs/PREREG_vlm_refusal_0919.md (실행 전 작성)",
        "_input": "텍스트만. 이미지는 검은 64x64 자리표시자",
        "_limit": "지시문을 어시스턴트가 작성했다. 제품 정확도로 인용 금지",
        "model": a.model, "variant": a.variant, "candidates": list(CANDIDATES),
        "arm_a_margin_abstain": {
            "threshold": thr, "percentile": ABSTAIN_PERCENTILE,
            "threshold_from": "지원 지시 중 맞힌 항목의 margin. 거절 집합 미사용",
            "support_hits": a_hits, "support_n": len(sup_rows), "support_acc": a_acc,
            "support_ci": [aa_lo, aa_hi],
            "support_abstained": sum(r["abstain"] for r in sup_rows),
            "refuse_refused": a_ref, "refuse_n": len(ref_rows),
            "refuse_ci": [ar_lo, ar_hi], "verdict": "채택" if a_pass else "미채택"},
        "support": {"hits": hits, "n": len(sup_rows), "acc": acc,
                    "ci": [a_lo, a_hi], "leaked_to_refuse": leak, "rows": sup_rows},
        "refuse": {"refused": refused, "n": len(ref_rows), "rate": rr,
                   "ci": [r_lo, r_hi], "by_bucket": split_counts(ref_rows),
                   "rows": ref_rows},
        "gate": {"G1": "0.83 CI [0.74, 0.89] 안", "G2": GATE_G2},
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
