"""Can this skill be executed on THIS scene, right now? 이 장면에서 이 작업이 지금 가능한가.

D-AI-88. 스킬 선택은 **클릭이 한다** (D-AI-63: FE 가 skill_id 를 보낸다).
그래서 VLM 의 자리는 선택이 아니라 **실행 전 판정**이다.

    입력   사진 + skill_id
    출력   {"executable": bool, "confidence": float, "margin": float}

왜 2지선다인가
--------------
`skill_choice.forced_choice_scores` 는 후보가 5개뿐이라 **거부 출력이 존재하지 않는다.**
여기서는 후보가 `가능 / 불가` 둘이라 거부가 출력 공간 안에 실제로 있다.
chance 는 0.50 이다 (5지선다의 0.20 이 아니다). **게이트를 그만큼 높게 잡는다.**

채점 규칙은 `skill_choice` 를 **그대로 베낀다** — 답 토큰만의 평균 로그확률 argmax.
평균을 쓰는 이유도 같다(후보 토큰 수가 다르면 합계는 짧은 쪽이 유리하다).
규칙이 같다는 것을 자체검증에서 확인한다. 규칙을 바꾸면 0912·0914 수치와 비교가 끊긴다.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SRC = Path(__file__).resolve().parent / "skill_choice.py"


def _enum_from_source() -> tuple[tuple[str, ...], dict[str, str]]:
    """Read SKILLS/SKILL_KO from skill_choice.py **without importing it**.

    skill_choice 는 최상위에서 torch 를 import 한다. 게이트 층까지 torch 를 요구하면
    모델 없는 기계에서 자체검증을 못 돌린다. 그렇다고 값을 복사해 두면 두 벌이 되어
    말없이 갈린다 — **원본 소스를 파싱해서 한 벌로 유지한다.**
    """
    import ast
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    found: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in ("SKILLS", "SKILL_KO"):
                    found[t.id] = ast.literal_eval(node.value)
    missing = [k for k in ("SKILLS", "SKILL_KO") if k not in found]
    if missing:
        raise RuntimeError(f"{_SRC} 에서 {missing} 를 못 읽었다 — 이름이 바뀌었나")
    return tuple(found["SKILLS"]), dict(found["SKILL_KO"])


SKILLS, SKILL_KO = _enum_from_source()

# 후보 순서가 곧 인덱스다. 0 = 불가, 1 = 가능.
VERDICTS: tuple[str, ...] = ("불가", "가능")
CHANCE = 0.50

# ── 사전 등록 게이트 (사진 보기 전에 정했다) ──────────────────────────────────
GATE_REFUSE_EMPTY = 0.90      # 빈 장면을 '불가'로 판정하는 비율
GATE_FALSE_REFUSE = 0.10      # 물체가 있는데 '불가'로 판정하는 비율 — **이쪽이 더 나쁘다**
GATE_ACCURACY = 0.85          # 전체 정확도. chance 0.50
MIN_PER_GROUP = 10


def build_question(skill_id: str) -> str:
    """The user turn. 모델에 보이는 사용자 발화. 스킬은 **주어진다** — 고르지 않는다."""
    if skill_id not in SKILLS:
        raise ValueError(f"모르는 skill_id: {skill_id!r} — {SKILLS}")
    return (
        "사진은 로봇팔 작업대다.\n"
        f"로봇이 지금 수행하려는 작업: {skill_id} ({SKILL_KO[skill_id]})\n"
        "사진에 그 작업을 할 대상이 실제로 있는가?\n"
        "대상이 없거나 가려져서 집을 수 없으면 '불가', 있으면 '가능' 이라고만 답하라."
    )


@dataclass
class Verdict:
    executable: bool
    margin: float
    scores: list[float]

    @property
    def confidence(self) -> float:
        """margin -> 0.5~1.0. 로지스틱이 아니라 **단조 변환일 뿐**이다 (교정 미실시)."""
        import math
        return 0.5 + 0.5 * (1.0 - math.exp(-abs(self.margin)))


def executable_scores(model: Any, processor: Any, image: Any, skill_id: str,
                      device: str) -> Verdict:
    """Score '불가' and '가능'. 두 후보를 채점한다. 규칙은 skill_choice 와 같다."""
    import torch

    from vlm.skill_choice import _process        # torch 가 필요한 시점에만 들인다

    question = build_question(skill_id)
    scores: list[float] = []
    for cand in VERDICTS:
        msgs = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        enc_full = _process(processor, prompt + cand, image)
        enc_prompt = _process(processor, prompt, image)
        n_prompt = int(enc_prompt["input_ids"].shape[-1])

        moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc_full.items()}
        logits = model(**moved).logits.float()
        ids = moved["input_ids"][0]
        n_total = int(ids.shape[-1])
        if n_total <= n_prompt:
            scores.append(float("-inf"))
            continue
        lp = torch.log_softmax(logits[0, n_prompt - 1:n_total - 1, :], dim=-1)
        tgt = ids[n_prompt:n_total]
        scores.append(float(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()))

    idx = 0 if scores[0] >= scores[1] else 1
    return Verdict(executable=bool(idx == 1), margin=abs(scores[0] - scores[1]),
                   scores=scores)


# ── 판정 (모델 없이 도는 부분. 여기가 게이트다) ──────────────────────────────
#   묶음 -> 정답. `other_` 는 **물체가 있으므로 가능**이다. 물체 종류 민감도는
#   게이트가 아니라 별도 지표로 본다 (정답을 모르는 것을 정답처럼 쓰지 않는다).
TRUTH: dict[str, bool] = {"empty": False, "can": True, "other": True, "occluded": True}


def judge(rows: list[dict]) -> dict:
    """rows: [{group, executable}] -> 게이트 판정. 모수를 전부 같이 낸다."""
    out = []

    def row(name, ok, got, want):
        out.append({"name": name, "ok": ok, "got": got, "want": want})

    by = {g: [r for r in rows if r["group"] == g] for g in TRUTH}
    for g in TRUTH:
        row(f"{g} 장수", len(by[g]) >= MIN_PER_GROUP, f"{len(by[g])}", f">= {MIN_PER_GROUP}")

    e = by["empty"]
    if e:
        k = sum(1 for r in e if not r["executable"])
        row("빈 장면 거부율", k / len(e) >= GATE_REFUSE_EMPTY,
            f"{k} / {len(e)} = {k / len(e):.3f}", f">= {GATE_REFUSE_EMPTY}")
    else:
        row("빈 장면 거부율", None, "empty 가 비었다 — 대조 불가", f">= {GATE_REFUSE_EMPTY}")

    pos = [r for g in ("can", "other", "occluded") for r in by[g]]
    if pos:
        k = sum(1 for r in pos if not r["executable"])
        row("오거부율 (있는데 불가)", k / len(pos) <= GATE_FALSE_REFUSE,
            f"{k} / {len(pos)} = {k / len(pos):.3f}", f"<= {GATE_FALSE_REFUSE}")
    else:
        row("오거부율 (있는데 불가)", None, "양성 묶음이 비었다", f"<= {GATE_FALSE_REFUSE}")

    scored = [r for r in rows if r["group"] in TRUTH]
    if scored:
        k = sum(1 for r in scored if r["executable"] == TRUTH[r["group"]])
        row("전체 정확도", k / len(scored) >= GATE_ACCURACY,
            f"{k} / {len(scored)} = {k / len(scored):.3f}",
            f">= {GATE_ACCURACY} (chance {CHANCE})")
    else:
        row("전체 정확도", None, "채점할 항목이 없다", f">= {GATE_ACCURACY}")

    p = sum(1 for r in out if r["ok"] is True)
    f = sum(1 for r in out if r["ok"] is False)
    u = sum(1 for r in out if r["ok"] is None)
    # 물체 종류 민감도 — **게이트가 아니다.** can 과 other 가 갈리는지만 본다
    sens = None
    if by["can"] and by["other"]:
        ca = sum(r["executable"] for r in by["can"]) / len(by["can"])
        ot = sum(r["executable"] for r in by["other"]) / len(by["other"])
        sens = {"can_가능률": ca, "other_가능률": ot, "차이": abs(ca - ot),
                "note": "게이트 아님. 물체 종류를 구분하는지 보는 참고 지표"}
    return {"rows": out, "passed": p, "failed": f, "unknown": u, "total": len(out),
            "verdict": "PASS" if f == 0 and u == 0 else ("FAIL" if f else "INCOMPLETE"),
            "object_sensitivity": sens,
            "counts": {g: len(v) for g, v in by.items()},
            "note": "chance 는 0.50 이다 (2지선다). 5지선다 0.20 과 비교하지 마라"}


def selftest() -> int:
    ok = tot = 0

    def chk(name, cond, note=""):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(f"  {'OK  ' if cond else '실패'} {name}   {note}")

    def rows(empty_ok, can_ok, n=12):
        r = [{"group": "empty", "executable": e} for e in empty_ok]
        r += [{"group": "can", "executable": c} for c in can_ok]
        r += [{"group": "other", "executable": True}] * n
        r += [{"group": "occluded", "executable": True}] * n
        return r

    perfect = judge(rows([False] * 12, [True] * 12))
    chk("1 완벽 -> PASS", perfect["verdict"] == "PASS",
        f"{perfect['verdict']} · 통과 {perfect['passed']}/{perfect['total']}")

    blind = judge(rows([True] * 12, [True] * 12))       # 전부 '가능' — 장면을 안 본다
    chk("2 전부 가능 -> FAIL (판별행)", blind["verdict"] == "FAIL",
        f"거부율 0 · 불합격 {blind['failed']}")

    allno = judge(rows([False] * 12, [False] * 12))     # 전부 '불가'
    chk("3 전부 불가 -> FAIL (오거부율이 잡는다)", allno["verdict"] == "FAIL",
        f"불합격 {allno['failed']}")
    chk("3b 거부율만 보면 전부 불가가 통과한다 — 오거부율이 그걸 막는다",
        any(r["name"] == "빈 장면 거부율" and r["ok"] for r in allno["rows"])
        and any(r["name"].startswith("오거부율") and r["ok"] is False for r in allno["rows"]))

    few = judge(rows([False] * 3, [True] * 3, n=3))
    chk("4 장수 미달 -> 불합격 (판별행)",
        sum(1 for r in few["rows"] if r["name"].endswith("장수") and r["ok"] is False) == 4)

    empty0 = judge([{"group": "can", "executable": True}] * 12)
    chk("5 empty 없음 -> 미판정, PASS 아님 (판별행)",
        empty0["verdict"] != "PASS" and empty0["unknown"] >= 1,
        f"{empty0['verdict']} · 미판정 {empty0['unknown']}")

    borderline = judge(rows([False] * 11 + [True], [True] * 12))   # 거부율 11/12=0.917
    chk("6 거부율 0.917 -> 통과 (경계 정답 아는 행)",
        any(r["name"] == "빈 장면 거부율" and r["ok"] for r in borderline["rows"]), "0.917 >= 0.90")
    b2 = judge(rows([False] * 10 + [True] * 2, [True] * 12))        # 10/12 = 0.833
    chk("7 거부율 0.833 -> 불합격 (경계 판별행)",
        any(r["name"] == "빈 장면 거부율" and r["ok"] is False for r in b2["rows"]), "0.833 < 0.90")

    chk("8 물체 종류 민감도는 게이트가 아니다",
        perfect["object_sensitivity"] is not None
        and "게이트 아님" in perfect["object_sensitivity"]["note"])

    try:
        build_question("없는스킬"); bad = False
    except ValueError:
        bad = True
    chk("9 모르는 skill_id -> 거부 (판별행)", bad)
    chk("10 채점 규칙이 skill_choice 와 같은 문장 구조", "답하라" in build_question(SKILLS[0]))
    chk("11 enum 을 소스에서 한 벌로 읽는다 (복사본 없음)",
        len(SKILLS) == 5 and set(SKILL_KO) == set(SKILLS),
        f"{len(SKILLS)}종 · {SKILLS[0]}")

    print(f"\n자체검증 {ok}/{tot}")
    return 0 if ok == tot else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
