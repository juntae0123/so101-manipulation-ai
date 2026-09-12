"""M0 -- can the base VLM pick the right skill, with no fine-tuning?
M0 -- 파인튜닝 없는 베이스 VLM 이 스킬을 고를 수 있는가?

Why this exists / 왜 있는가.

D-AI-17 이 LoRA 착수 **전에** M0 를 재라고 못박았다. baseline 없이 파인튜닝을 돌리면
성능이 나와도 파인튜닝 덕인지 사전학습 덕인지 가릴 수 없다 (BC->ACT 규칙과 같은 논리).
게이트는 D-AI-17 에 이미 고정돼 있다. **여기서 숫자를 바꾸지 않는다.**

    n=100 · 5지선다 · 우연 = 0.20
    >= 0.90        파인튜닝 취소. 자원을 grounding 정확도로 돌린다
    0.50 ~ 0.90    LoRA 필요. 시뮬 렌더 데이터셋 진행
    <  0.50        모델 후보 재검토

두 가지를 따로 잰다 -- 섞으면 어느 쪽이 고장인지 모른다.

**(1) 강제선택 정확도**  후보 5개 각각을 답으로 붙여 평균 토큰 로그확률을 재고 argmax.
    항상 유효한 답이 나오므로 **형식 실패가 정확도를 오염시키지 않는다.**
    D-AI-17 게이트는 이 수치로 판정한다.

**(2) 자유생성 형식 준수율**  실제로 JSON 을 내게 하고 enum 밖 값·파싱 실패를 센다.
    D-AI-24(enum 제약 + 기권 게이트)가 필요한지 여부가 이 수치로 갈린다.

그리고 **빈 이미지 조건을 반드시 함께 돌린다.** 지시문만으로 답이 나오면 이 측정은
비전 능력을 잰 것이 아니다. 관측 절제(2026-09-11)에서 쓴 것과 같은 방법이다 --
조건을 지우고 얼마나 떨어지는지를 본다. 빈 이미지와 성능이 같으면 **M0 는
'지시문 분류' 를 잰 것이고 비전은 기여하지 않는다**고 읽어야 한다.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# D-AI-17 의 게이트. 결과를 보기 전에 확정됐다 -- 결과에 맞춰 고치면 판정이 아니다.
GATE_SKIP_FINETUNE = 0.90
GATE_NEED_LORA = 0.50
CHANCE = 0.20

# D-AI-22 의 스킬 5종. enum 이 정본이고 순서가 곧 인덱스다.
SKILLS: tuple[str, ...] = (
    "pick_place",
    "sort_two",
    "align_fixture",
    "present_inspect",
    "line_up",
)

SKILL_KO: dict[str, str] = {
    "pick_place": "집어 옮기기",
    "sort_two": "두 곳으로 분류",
    "align_fixture": "지그에 정렬",
    "present_inspect": "검사 자세 제시",
    "line_up": "순서대로 늘어놓기",
}

# 지시문 100개 = 스킬당 20개.
#
# ⚠️ **이 문장들은 어시스턴트가 썼다. 실사용자 발화가 아니다.** 판정하는 쪽과 문제를
#    내는 쪽이 같으므로, 실제 사용자 말투보다 쉬울 수 있다. M0 정확도를 "사용자가 말하면
#    이만큼 맞춘다"로 읽지 않는다. 사람이 다시 쓰면 그 판이 정본이 된다.
# 원칙: 스킬 이름을 그대로 부르지 않는다(그러면 문자열 매칭 문제가 된다).
#       구어체·존댓말·간접 지시를 섞는다. 물체 이름은 일부러 다양하게 둔다.
INSTRUCTIONS: dict[str, tuple[str, ...]] = {
    "pick_place": (
        "저거 집어서 트레이에 옮겨줘",
        "부품 하나 들어서 옆 통에 넣어",
        "이거 집어다가 저쪽에 놔줘",
        "가운데 있는 거 들어서 상자에 담아",
        "그 물건 옮겨 담아주세요",
        "집어서 반대편으로 보내줘",
        "노란 상자 들어다가 트레이에 두면 돼",
        "하나 집어서 자리 옮겨",
        "저 물건을 저 통으로 이동시켜",
        "테이블 위의 것 하나 집어서 옮겨줘요",
        "그거 들어서 비어 있는 칸에 넣어",
        "부품을 다른 자리로 옮기기만 하면 된다",
        "집어 올려서 옆으로 내려놔",
        "이 물체 트레이로 이송해줘",
        "집어서 지정된 위치에 두세요",
        "물건 하나 골라서 자리 바꿔줘",
        "저기 있는 걸 여기로 가져다 놔",
        "픽업해서 놓는 것만 해줘",
        "하나 들어서 옆 자리에 내려놓기",
        "그 부품 집어 다른 데로 옮겨라",
    ),
    "sort_two": (
        "불량이면 오른쪽, 정상이면 왼쪽 통에 넣어",
        "긁힘 있는 것만 따로 골라내줘",
        "두 종류로 나눠서 담아",
        "괜찮은 거랑 아닌 거 구분해서 넣어줘",
        "색깔별로 두 칸에 나눠 담아",
        "합격품과 불합격품을 갈라놔",
        "이상 있는 건 오른쪽으로 빼",
        "검사해서 통과한 것만 왼쪽에 모아줘",
        "둘로 분류해서 각각 트레이에 넣어",
        "정상품은 그대로, 불량품은 따로 빼주세요",
        "크기 큰 건 이쪽 작은 건 저쪽",
        "판별해서 해당하는 통에 넣어",
        "종류 보고 맞는 쪽에 담아줘",
        "쓸 수 있는 것과 버릴 것을 나눠",
        "두 개 통 중에 골라서 넣는 거야",
        "선별해서 각기 다른 자리에 배치해",
        "하자 있는 건 따로 모아둬",
        "양품 불량 구분 작업 해줘",
        "조건에 따라 좌우로 갈라 담아",
        "보고 판단해서 둘 중 한쪽에 넣어라",
    ),
    "align_fixture": (
        "지그에 맞춰서 넣어줘",
        "치구 홈에 정확히 끼워",
        "고정대에 자세 맞춰 올려놔",
        "거치대에 방향 맞춰서 세워줘",
        "홀더에 딱 맞게 넣어주세요",
        "기구물에 정렬해서 안착시켜",
        "정해진 자리에 각도까지 맞춰 놓기",
        "받침에 똑바로 맞춰 끼워넣어",
        "가이드에 맞게 위치 잡아줘",
        "틀에 정확히 들어가게 넣어",
        "지그 방향대로 회전시켜서 놔",
        "정렬해서 고정 위치에 올려",
        "홈에 맞게 자세 잡아 삽입해",
        "픽스처에 정합시켜줘",
        "각 맞춰서 거치해주세요",
        "자리에 딱 떨어지게 맞춰 넣어",
        "치구에 세팅해",
        "정해진 방향으로 돌려서 안착",
        "고정 지그에 물려줘",
        "틀어지지 않게 맞춰 넣어라",
    ),
    "present_inspect": (
        "카메라 앞에 들어서 보여줘",
        "잘 보이게 들어올려봐",
        "검사할 수 있게 앞으로 내밀어",
        "돌려가면서 보여줄래",
        "이거 표면 확인하게 들어줘",
        "육안 검사 자세로 제시해",
        "들어서 한 바퀴 돌려봐",
        "확인할 수 있게 앞에 갖다 대",
        "살펴볼 수 있게 위치시켜주세요",
        "보여주고 다시 내려놔",
        "검사용으로 들어 보여줘",
        "앞면 뒷면 다 보이게 회전시켜",
        "확대해서 볼 수 있게 가까이 들어",
        "제시 자세 취해줘",
        "판독할 수 있게 각도 틀어서 보여줘",
        "들어서 보여준 다음 제자리에",
        "검수하게 앞으로 가져와",
        "상태 볼 수 있게 위로 들어",
        "이거 보여주기만 하면 돼",
        "검사 포즈로 들어올려라",
    ),
    "line_up": (
        "일렬로 줄 세워",
        "차례대로 나란히 놓아줘",
        "순서에 맞게 늘어놓아",
        "하나씩 옆으로 붙여서 배열해",
        "가지런히 한 줄로 정리해줘",
        "번호순으로 쭉 놓아주세요",
        "간격 맞춰서 일자로 배치해",
        "앞에서부터 순서대로 깔아",
        "줄 맞춰서 정렬해놔",
        "쭉 늘어놓기만 하면 돼",
        "순번대로 한 줄 만들어",
        "옆으로 이어서 나열해줘",
        "일정한 간격으로 줄지어 놔",
        "차례차례 늘어놓는 작업",
        "라인으로 배열해주세요",
        "하나 놓고 그 옆에 또 놓고 반복",
        "정해진 순서로 쭉 깔아놔",
        "줄 세우기 해줘",
        "순서 맞춰 일렬 배치",
        "차례대로 한 줄로 세워라",
    ),
}

# enum 설명 v2 — M0(2026-09-12) 이후. **원 측정을 대체하지 않는다. 별도 조건이다.**
#
# v1 실측 🟢: `sort_two` 0.35 이고 오분류 12/20 이 **전부 `pick_place` 방향**이었다.
# v1 의 "집어 옮기기"는 나머지 넷의 상위 개념으로 읽힌다 — 분류도 정렬도 집어 옮기는 일이다.
# v2 는 **서로를 배제하는 특징**을 각 항목에 넣는다. 파인튜닝 전에 이것부터 재는 이유:
# 프롬프트로 고쳐지는 것을 가중치로 고치면 무엇이 고쳐졌는지 귀속시킬 수 없다.
SKILL_KO_V2: dict[str, str] = {
    "pick_place": "한 물체를 집어 **미리 정해진 한 곳**으로 옮긴다. 판별도 자세 맞춤도 없다",
    "sort_two": "물체를 **보고 판별해 두 목적지 중 하나를 고른다**. 목적지가 물체마다 달라진다",
    "align_fixture": "지그·치구의 정해진 자리에 **방향과 각도를 맞춰** 끼운다",
    "present_inspect": "들어서 **보여준 뒤 제자리로 돌아온다**. 다른 곳으로 옮기지 않는다",
    "line_up": "**여러 개를** 차례대로 간격을 맞춰 한 줄로 늘어놓는다",
}

_ENUM_LINE = " | ".join(SKILLS)

def system_task(variant: str = "v1") -> str:
    """The enum block shown to the model. `variant` picks the description set.
    모델에 보이는 enum 블록. `variant` 가 설명 판을 고른다."""
    table = SKILL_KO if variant == "v1" else SKILL_KO_V2
    return (
        "로봇팔이 수행할 스킬을 하나 고른다.\n"
        f"가능한 값은 다음 다섯 개뿐이다: {_ENUM_LINE}\n"
        + "\n".join(f"- {s}: {table[s]}" for s in SKILLS)
    )


SYSTEM_TASK = system_task("v1")


def build_question(instruction: str, variant: str = "v1") -> str:
    """The user turn shown to the model.
    모델에 보이는 사용자 발화."""
    return (
        f"{system_task(variant)}\n\n"
        f'사진은 작업 현장이다. 지시: "{instruction}"\n'
        "이 지시에 해당하는 skill_id 하나만 답하라."
    )


def build_json_question(instruction: str, variant: str = "v1") -> str:
    """Free-generation variant -- measures D-AI-24 format adherence.
    자유생성용 -- D-AI-24 형식 준수율을 잰다."""
    return (
        f"{system_task(variant)}\n\n"
        f'사진은 작업 현장이다. 지시: "{instruction}"\n'
        'JSON 한 줄로만 답하라: {"skill_id": "...", "confidence": 0.0, "abstain": false}'
    )


@dataclass
class Item:
    """One test item. Label comes from which bucket the instruction was written in.
    검사 항목 하나. 라벨은 지시문이 어느 묶음에서 왔는지로 정해진다."""

    index: int
    skill: str
    instruction: str


def build_items() -> list[Item]:
    """The fixed 100. Order is deterministic; the set never depends on results.
    고정된 100개. 순서는 결정적이고, 집합이 결과에 의존하지 않는다."""
    items: list[Item] = []
    for skill in SKILLS:
        for text in INSTRUCTIONS[skill]:
            items.append(Item(index=len(items), skill=skill, instruction=text))
    return items


def check_items(items: list[Item]) -> list[str]:
    """Refuse to run on a malformed item set. A broken instrument stays silent.
    잘못된 항목 집합이면 실행을 거부한다. 고장난 계측기는 스스로 알리지 않는다."""
    problems: list[str] = []
    if len(items) != 100:
        problems.append(f"항목이 100개가 아니다: {len(items)}개")
    counts = {s: sum(1 for it in items if it.skill == s) for s in SKILLS}
    if set(counts.values()) != {20}:
        problems.append(f"스킬별 균형이 깨졌다: {counts} (전부 20 이어야 한다)")
    texts = [it.instruction for it in items]
    if len(set(texts)) != len(texts):
        dup = sorted({t for t in texts if texts.count(t) > 1})
        problems.append(f"중복 지시문 {len(dup)}건: {dup[:5]}")
    # 스킬 이름이 지시문에 그대로 들어 있으면 문자열 매칭 문제가 된다.
    for it in items:
        low = it.instruction.lower()
        for s in SKILLS:
            if s in low or s.replace("_", " ") in low:
                problems.append(f"지시문에 스킬 이름이 노출됐다: [{it.index}] {it.instruction!r}")
    return problems


# ----------------------------------------------------------------------------- 채점


_JSON_RE = re.compile(r"\{.*?\}", re.S)


@dataclass
class FreeGenResult:
    """What free generation produced for one item.
    한 항목의 자유생성 결과."""

    raw: str
    parsed_json: bool = False
    skill_in_enum: bool = False
    skill: str | None = None
    abstain: bool | None = None


def parse_free_gen(text: str) -> FreeGenResult:
    """Parse the model's free-form answer without being generous about it.
    모델의 자유 답변을 관대하지 않게 파싱한다.

    관대한 파서는 형식 결함을 숨긴다 -- 그러면 D-AI-24 가 필요한지 알 수 없게 된다.
    JSON 으로 안 나오면 그냥 실패로 센다.
    """
    res = FreeGenResult(raw=text)
    m = _JSON_RE.search(text)
    if m is None:
        return res
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001 - 깨진 JSON 도 실패다
        return res
    if not isinstance(obj, dict):
        return res
    res.parsed_json = True
    sid = obj.get("skill_id")
    if isinstance(sid, str):
        res.skill = sid
        res.skill_in_enum = sid in SKILLS
    ab = obj.get("abstain")
    res.abstain = ab if isinstance(ab, bool) else None
    return res


@dataclass
class ConditionReport:
    """One (model, image condition) cell.
    (모델, 이미지 조건) 한 칸."""

    model_id: str
    condition: str
    n: int
    forced_correct: int
    forced_acc: float
    per_skill_acc: dict[str, float]
    confusion: list[list[int]]
    json_parsed: int
    enum_ok: int
    freegen_correct: int
    abstain_count: int
    mean_margin: float = 0.0
    notes: list[str] = field(default_factory=list)

    def verdict(self) -> str:
        """D-AI-17 게이트를 그대로 적용한다. 여기서 기준을 만들지 않는다."""
        a = self.forced_acc
        if a >= GATE_SKIP_FINETUNE:
            return "파인튜닝 취소 구간 (>=0.90) -- 자원을 grounding 으로"
        if a >= GATE_NEED_LORA:
            return "LoRA 필요 구간 (0.50~0.90)"
        return "모델 후보 재검토 구간 (<0.50)"


def score_condition(
    model_id: str,
    condition: str,
    items: list[Item],
    forced_choice: list[int],
    margins: list[float],
    free_texts: list[str] | None,
) -> ConditionReport:
    """Turn raw per-item outputs into the numbers we report.
    항목별 원시 출력을 보고할 수치로 바꾼다."""
    idx = {s: i for i, s in enumerate(SKILLS)}
    conf = [[0] * len(SKILLS) for _ in SKILLS]
    correct = 0
    per_skill_hit = {s: 0 for s in SKILLS}
    per_skill_tot = {s: 0 for s in SKILLS}

    for it, pick in zip(items, forced_choice):
        conf[idx[it.skill]][pick] += 1
        per_skill_tot[it.skill] += 1
        if SKILLS[pick] == it.skill:
            correct += 1
            per_skill_hit[it.skill] += 1

    json_parsed = enum_ok = freegen_correct = abstain_count = 0
    if free_texts is not None:
        for it, raw in zip(items, free_texts):
            r = parse_free_gen(raw)
            json_parsed += int(r.parsed_json)
            enum_ok += int(r.skill_in_enum)
            freegen_correct += int(r.skill == it.skill)
            abstain_count += int(bool(r.abstain))

    return ConditionReport(
        model_id=model_id,
        condition=condition,
        n=len(items),
        forced_correct=correct,
        forced_acc=correct / len(items) if items else 0.0,
        per_skill_acc={s: (per_skill_hit[s] / per_skill_tot[s] if per_skill_tot[s] else 0.0)
                       for s in SKILLS},
        confusion=conf,
        json_parsed=json_parsed,
        enum_ok=enum_ok,
        freegen_correct=freegen_correct,
        abstain_count=abstain_count,
        mean_margin=float(np.mean(margins)) if margins else 0.0,
    )


def wilson95(k: int, n: int) -> tuple[float, float]:
    """95% interval for a proportion. n=100 에서 반폭이 약 +-9%p 라는 사실을 눈에 보이게 한다."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


@torch.no_grad()
def forced_choice_scores(
    model: Any,
    processor: Any,
    image: Any,
    instruction: str,
    device: str,
    variant: str = "v1",
) -> tuple[int, float, list[float]]:
    """Score all five candidates and return (argmax, margin, per-candidate mean logprob).

    다섯 후보를 각각 정답으로 붙여 **답 토큰만의 평균 로그확률**을 잰다.
    평균을 쓰는 이유: 후보마다 토큰 수가 달라 합계를 쓰면 짧은 이름이 유리해진다.
    이 규칙은 결과를 보기 전에 정했다.
    """
    question = build_question(instruction, variant)
    scores: list[float] = []

    for cand in SKILLS:
        msgs = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        full = prompt + cand

        enc_full = _process(processor, full, image)
        enc_prompt = _process(processor, prompt, image)
        n_prompt = int(enc_prompt["input_ids"].shape[-1])

        moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc_full.items()}
        logits = model(**moved).logits.float()

        ids = moved["input_ids"][0]
        n_total = int(ids.shape[-1])
        if n_total <= n_prompt:
            scores.append(float("-inf"))
            continue
        # 위치 t 의 로짓이 토큰 t+1 을 예측한다.
        lp = torch.log_softmax(logits[0, n_prompt - 1:n_total - 1, :], dim=-1)
        tgt = ids[n_prompt:n_total]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        scores.append(float(tok_lp.mean()))

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    margin = scores[order[0]] - scores[order[1]]
    return order[0], margin, scores


def _process(processor: Any, text: str, image: Any) -> dict[str, Any]:
    """Processor calling conventions differ between model families -- try them in order.
    프로세서 호출 관례가 모델 계열마다 달라서 순서대로 시도한다.
    (vlm/fp16_safety.py 에서 서버 실행으로 검증된 것과 같은 방식이다.)"""
    last: Exception | None = None
    for kwargs in (
        {"text": [text], "images": [image]},
        {"text": [text], "images": [[image]]},
        {"text": text, "images": image},
    ):
        try:
            return processor(return_tensors="pt", **kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"프로세서 입력 구성 실패: {type(last).__name__} {last}")
