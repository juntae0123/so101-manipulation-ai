"""M0 -- base VLM skill-selection accuracy, with a blank-image control.
M0 -- 파인튜닝 없는 베이스 VLM 의 스킬 선택 정확도. 빈 이미지 대조군 포함.

사전등록: docs/PREREG_vlm_m0_0912.md
결정: D-AI-17 (게이트 고정) · D-AI-24 (출력 형식) · 이슈 S15P21A103-36, -112

무엇을 하는가:
  조건마다 100항목을 돌린다. 항목은 (이미지, 한국어 지시문, 정답 스킬) 이고,
  스킬 5종 균형 20/20/20/20/20 이다.

    forced   후보 5개의 평균 토큰 로그확률 -> argmax. **D-AI-17 게이트는 이 수치로 판정**
    freegen  JSON 을 내게 하고 파싱 성공률 · enum 준수율을 센다 (D-AI-24 필요성)

이미지 조건:
    blank    0으로 채운 이미지. **대조군.** 이것과 성능이 같으면 비전이 기여하지 않는다
    render   MuJoCo 렌더 (계약 에피소드 .npz 의 cam_wrist)
    real     실물 시연 프레임 (UMI 번들 jpg)

blank 는 항상 돈다. 빼지 않는다 -- 빼면 "지시문 분류"를 "VLM 성능"으로 읽게 된다.

실행 예:
    MUJOCO_GL=egl PYTHONPATH=$PWD python tools/probe_vlm_m0.py \
        --models HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
        --render-dir datasets/sim_pick_v3 --log
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm.fp16_safety import _load_model  # noqa: E402  서버 실행으로 검증된 로더를 재사용한다
from vlm.skill_choice import (  # noqa: E402
    CHANCE,
    SKILLS,
    ConditionReport,
    build_items,
    build_json_question,
    check_items,
    forced_choice_scores,
    score_condition,
    wilson95,
    _process,
)
from tracking.exp_log import log_run  # noqa: E402

IMAGE_SIZE = 224


def _blank_images(n: int) -> list[Any]:
    from PIL import Image
    return [Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))] * n


def _render_images(root: Path, n: int, cam: str = "cam_wrist") -> list[Any]:
    """One frame per contract episode, taken at a fixed relative tick.
    계약 에피소드마다 한 장. 고정된 상대 틱에서 뽑아 결정적이다."""
    from PIL import Image
    from contract.episode import read_episode

    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"렌더 에피소드가 없다: {root}")
    out: list[Any] = []
    i = 0
    while len(out) < n:
        ep = read_episode(files[i % len(files)])
        if cam not in ep.images:
            raise KeyError(f"{files[i % len(files)].name} 에 카메라 {cam} 가 없다 "
                           f"(있는 것: {sorted(ep.images)})")
        arr = ep.images[cam]            # (T,3,H,W) uint8
        t = (len(arr) * (len(out) * 7 + 3) // 10) % len(arr)   # 결정적으로 흩뿌린다
        out.append(Image.fromarray(np.transpose(arr[t], (1, 2, 0))).resize(
            (IMAGE_SIZE, IMAGE_SIZE)))
        i += 1
    return out


def _real_images(root: Path, n: int) -> list[Any]:
    """One frame per UMI episode folder, at a fixed relative position.
    UMI 에피소드 폴더마다 한 장. 상대 위치 고정."""
    from PIL import Image

    eps = sorted(p for p in root.glob("rec_*") if (p / "frames").is_dir())
    if not eps:
        raise FileNotFoundError(f"실물 에피소드가 없다: {root}")
    out: list[Any] = []
    i = 0
    while len(out) < n:
        frames = sorted((eps[i % len(eps)] / "frames").glob("*.jpg"))
        if frames:
            t = (len(frames) * (len(out) * 7 + 3) // 10) % len(frames)
            img = Image.open(frames[t]).convert("RGB")
            # 폰이 거꾸로 장착돼 프레임이 180도 돌아 있다 (2026-09-11 실측 🟢).
            # 여기서 세워둔다. 로봇팔 카메라 방향이 확정되면 이 줄을 다시 본다.
            out.append(img.rotate(180).resize((IMAGE_SIZE, IMAGE_SIZE)))
        i += 1
        if i > len(eps) * 4:
            raise RuntimeError(f"프레임을 {len(out)}장밖에 못 모았다 (필요 {n})")
    return out


@torch.no_grad()
def _free_generate(model: Any, processor: Any, image: Any, instruction: str,
                   device: str, max_new_tokens: int, variant: str = "v1") -> str:
    msgs = [{"role": "user",
             "content": [{"type": "image"},
                         {"type": "text", "text": build_json_question(instruction, variant)}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    enc = _process(processor, prompt, image)
    moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    n_in = int(moved["input_ids"].shape[-1])
    out = model.generate(**moved, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]


def run_condition(model: Any, processor: Any, model_id: str, condition: str,
                  items: list, images: list[Any], device: str,
                  do_freegen: bool, max_new_tokens: int,
                  variant: str = "v1") -> ConditionReport:
    picks: list[int] = []
    margins: list[float] = []
    frees: list[str] | None = [] if do_freegen else None
    t0 = time.time()
    for k, (it, img) in enumerate(zip(items, images)):
        pick, margin, _ = forced_choice_scores(model, processor, img, it.instruction,
                                              device, variant)
        picks.append(pick)
        margins.append(margin)
        if frees is not None:
            frees.append(_free_generate(model, processor, img, it.instruction,
                                        device, max_new_tokens, variant))
        if (k + 1) % 20 == 0:
            done = k + 1
            print(f"    {condition}: {done}/{len(items)}  "
                  f"({(time.time() - t0) / done:.2f}s/항목)", flush=True)
    return score_condition(model_id, condition, items, picks, margins, frees)


def format_report(reports: list[ConditionReport], variant: str = "v1") -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"M0 — 베이스 VLM 스킬 선택 (파인튜닝 없음) · enum 설명판 {variant}")
    lines.append(f"우연 = {CHANCE:.2f} · n=100 · 95% 구간 반폭 약 ±9%p")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"{'모델':38s} {'조건':8s} {'강제선택':>9s} {'95% 구간':>16s} "
                 f"{'JSON':>6s} {'enum':>6s} {'margin':>8s}")
    for r in reports:
        lo, hi = wilson95(r.forced_correct, r.n)
        lines.append(f"{r.model_id.split('/')[-1][:38]:38s} {r.condition:8s} "
                     f"{r.forced_acc:9.2f} {f'[{lo:.2f}, {hi:.2f}]':>16s} "
                     f"{r.json_parsed:6d} {r.enum_ok:6d} {r.mean_margin:8.3f}")
    lines.append("")

    by_model: dict[str, dict[str, ConditionReport]] = {}
    for r in reports:
        by_model.setdefault(r.model_id, {})[r.condition] = r

    for mid, cond in by_model.items():
        lines.append("-" * 78)
        lines.append(mid)
        blank = cond.get("blank")
        for name, r in cond.items():
            if name == "blank":
                continue
            if blank is None:
                continue
            d = r.forced_acc - blank.forced_acc
            lines.append(f"  {name} − blank = {d:+.2f}  "
                         + ("→ 이미지가 기여한다" if d >= 0.10 else
                            "→ **이미지 기여 미확인.** 지시문만으로 같은 성능이 나온다"))
        best = max(cond.values(), key=lambda r: r.forced_acc)
        lines.append(f"  최고 조건 {best.condition} {best.forced_acc:.2f} → {best.verdict()}")
        lines.append("  스킬별 정확도 (" + best.condition + "):")
        for s in SKILLS:
            lines.append(f"    {s:18s} {best.per_skill_acc[s]:.2f}")
        lines.append("  혼동행렬 (행=정답, 열=예측, " + best.condition + "):")
        lines.append("           " + " ".join(f"{s[:8]:>8s}" for s in SKILLS))
        for i, s in enumerate(SKILLS):
            lines.append(f"    {s[:8]:>8s} " + " ".join(f"{v:8d}" for v in best.confusion[i]))
    lines.append("")
    lines.append("⚠️ 지시문 100개는 어시스턴트가 썼다. 실사용자 발화가 아니다.")
    lines.append("⚠️ blank 대조군과 차이가 작으면 이 수치는 '지시문 분류' 정확도다.")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True,
                   help="HF 모델 id. fp16 안전성 통과한 것만 쓴다 (MEASURE_vlm_fp16_0901)")
    p.add_argument("--render-dir", type=Path, default=None,
                   help="계약 에피소드 .npz 디렉터리. 생략하면 render 조건을 건너뛴다")
    p.add_argument("--real-dir", type=Path, default=None,
                   help="UMI 번들 rec_* 상위 폴더. 생략하면 real 조건을 건너뛴다")
    p.add_argument("--camera", type=str, default="cam_wrist",
                   help="렌더 조건에서 쓸 카메라 (실물에 대응물이 있는 것은 손목뿐이다. L76)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-freegen", action="store_true",
                   help="자유생성을 건너뛴다 (강제선택만). 빠르지만 D-AI-24 수치가 안 나온다")
    p.add_argument("--prompt-variant", choices=("v1", "v2"), default="v1",
                   help="enum 설명 판. v2 는 서로를 배제하는 특징을 넣은 것 "
                        "(PREREG_vlm_m0_prompt_v2_0912.md). **v1 측정을 대체하지 않는다**")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--out", type=Path, default=None, help="결과 JSON 경로")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    items = build_items()
    problems = check_items(items)
    if problems:
        print("✗ 항목 집합이 잘못됐다. 해석하지 않는다:")
        for x in problems:
            print("   " + x)
        return 2
    print(f"항목 {len(items)}개 · 스킬 {len(SKILLS)}종 균형 확인 🟢")

    conditions: dict[str, list[Any]] = {"blank": _blank_images(len(items))}
    if args.render_dir is not None:
        conditions["render"] = _render_images(args.render_dir, len(items), args.camera)
        print(f"render: {args.render_dir} 에서 {len(items)}장")
    if args.real_dir is not None:
        conditions["real"] = _real_images(args.real_dir, len(items))
        print(f"real:   {args.real_dir} 에서 {len(items)}장 (180도 회전 적용)")
    if len(conditions) == 1:
        print("⚠️ blank 뿐이다. 이미지 조건이 없으면 M0 판정을 쓸 수 없다 — "
              "--render-dir 또는 --real-dir 을 준다")

    from transformers import AutoProcessor

    reports: list[ConditionReport] = []
    for model_id in args.models:
        print(f"\n=== {model_id} ===", flush=True)
        processor = AutoProcessor.from_pretrained(model_id)
        # V100(sm_70) 은 bf16 연산이 없다. fp16 안전성은 MEASURE_vlm_fp16_0901 에서 확인됐다.
        model = _load_model(model_id, torch.float16, args.device)
        for cname, imgs in conditions.items():
            r = run_condition(model, processor, model_id, cname, items, imgs,
                              args.device, not args.no_freegen, args.max_new_tokens,
                              args.prompt_variant)
            lo, hi = wilson95(r.forced_correct, r.n)
            print(f"  {cname:8s} 강제선택 {r.forced_acc:.2f} [{lo:.2f},{hi:.2f}] · "
                  f"JSON {r.json_parsed}/100 · enum {r.enum_ok}/100", flush=True)
            reports.append(r)
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    text = format_report(reports, args.prompt_variant)
    print("\n" + text)

    payload = {
        "items": len(items),
        "chance": CHANCE,
        "conditions": sorted(conditions),
        "reports": [
            {k: v for k, v in r.__dict__.items()} for r in reports
        ],
    }
    out = args.out or Path("out") / (
        f"vlm_m0_{args.prompt_variant}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과: {out}")

    if args.log:
        log_run(
            "vlm_m0",
            author="김준태(트랙B)",
            issue="S15P21A103-36",
            conditions={
                "models": args.models,
                "image_conditions": sorted(conditions),
                "camera": args.camera,
                "freegen": not args.no_freegen,
                "prompt_variant": args.prompt_variant,
                "prereg": "docs/PREREG_vlm_m0_0912.md",
            },
            result={
                f"{r.model_id.split('/')[-1]}__{r.condition}": {
                    "forced_acc": r.forced_acc,
                    "json_parsed": r.json_parsed,
                    "enum_ok": r.enum_ok,
                    "mean_margin": r.mean_margin,
                }
                for r in reports
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
