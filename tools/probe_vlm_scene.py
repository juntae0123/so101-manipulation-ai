#!/usr/bin/env python3
"""Does the VLM's answer depend on what is actually on the table?
VLM 의 답이 **책상 위에 무엇이 있는지**에 실제로 의존하는가.

사전등록: AI/docs/PREREG_vlm_onsite_0921.md (사진 도착 **전에** 작성)
이슈: S15P21A103-36, -109

먼저 적어두는 구조적 사실 🟢
-----------------------------
현재 선택 경로 `vlm/skill_choice.forced_choice_scores` 는 **강제선택**이다.
후보가 `SKILLS` 5개뿐이라 **"거부"라는 출력이 존재하지 않는다.**
→ "빈 장면에서 거부하는가" 를 그대로 물으면 답은 재볼 것도 없이 0% 다.
  **"재봤더니 0" 과 "낼 수 없어서 0" 은 다른 상태다.**

그래서 이 도구가 재는 것은 거부율이 아니라 **의존성**이다:

    같은 지시문 · 다른 사진 -> 점수가 달라지는가?

    안 달라진다  =>  VLM 이 이미지를 안 본다. 스킬 선택은 지시문 분류일 뿐이다.
                     이 경우 빈 장면 거부는 **구조적으로 불가능**하고, 거부를 원하면
                     후보를 늘리거나(probe_vlm_refusal) 별도 판정기를 둬야 한다.
    달라진다    =>  이미지가 기여한다. 그때 비로소 거부 임계를 논할 수 있다.

사진 묶음 (파일명 접두어)
-------------------------
    empty_*      빈 작업대
    can_*        대상물 있음
    other_*      비슷한 크기·색의 **다른** 물체        <- 판별행
    occluded_*   대상물이 절반쯤 가려짐

이 계측기가 스스로를 못 믿는 지점
----------------------------------
1. **빈 장면 검정** — empty 와 can 사진의 평균 절대 화소차가 임계 미만이면 두 묶음이
   사실 같은 그림이다. 그러면 "차이 없음"이 **공짜로** 나온다. 그 경우 죽는다.
2. **묶음 모수를 반드시 찍는다** — 읽은 파일 / 묶인 파일 / 버린 파일.
3. **한 묶음이라도 비면 판정하지 않는다.** 없음과 괜찮음을 가른다.

Usage
    python probe_vlm_scene.py --selftest
    python probe_vlm_scene.py --photos ~/data/vlm_onsite --model <hf-id> --out out/vlm_scene.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GROUPS = ("empty", "can", "other", "occluded")
IMG_EXT = (".jpg", ".jpeg", ".png")

# ── 사전 등록 게이트 (사진 보기 전에 정했다. 결과 보고 옮기지 않는다) ──────────
PIXEL_DIFF_MIN = 2.0        # empty vs can 평균 |화소차| (0~255). 미만이면 같은 그림이다
MIN_PER_GROUP = 10          # 묶음당 최소 장수. 미만이면 판정하지 않는다
DEP_MARGIN_MIN = 0.02       # empty·can 의 margin 중앙값 차이가 이 미만이면 "의존성 없음"
DEP_ARGMAX_MIN = 0.10       # argmax 분포가 이만큼도 안 달라지면 "의존성 없음"


def collect(photos: Path) -> tuple[dict[str, list[Path]], dict]:
    """Group photos by filename prefix. 접두어로 묶는다. **모수를 같이 낸다.**"""
    groups: dict[str, list[Path]] = {g: [] for g in GROUPS}
    seen = dropped = 0
    for p in sorted(photos.iterdir()) if photos.is_dir() else []:
        if p.suffix.lower() not in IMG_EXT:
            continue
        seen += 1
        for g in GROUPS:
            if p.name.startswith(g + "_"):
                groups[g].append(p)
                break
        else:
            dropped += 1
    tally = {"읽은 파일": seen, "묶인 파일": seen - dropped, "접두어 불명": dropped,
             **{g: len(v) for g, v in groups.items()}}
    return groups, tally


def load_gray(p: Path):
    """Grayscale array for the blank test. 빈 장면 검정용 회색조 배열."""
    import numpy as np
    try:
        from PIL import Image
        return np.asarray(Image.open(p).convert("L"), dtype=np.float64)
    except ImportError:
        import cv2
        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if a is None:
            raise RuntimeError(f"이미지를 못 읽었다: {p}")
        return np.asarray(a, dtype=np.float64)


def blank_test(groups: dict[str, list[Path]]) -> dict:
    """Are 'empty' and 'can' actually different pictures? 정말 다른 그림인가."""
    import numpy as np
    a, b = groups["empty"], groups["can"]
    if not a or not b:
        return {"ok": None, "why": f"empty {len(a)}장 · can {len(b)}장 — 한쪽이 비었다"}
    n = min(len(a), len(b), 10)
    diffs = []
    for pa, pb in zip(a[:n], b[:n]):
        ga, gb = load_gray(pa), load_gray(pb)
        if ga.shape != gb.shape:
            return {"ok": None, "why": f"해상도가 다르다 {ga.shape} vs {gb.shape}"}
        diffs.append(float(np.abs(ga - gb).mean()))
    m = float(np.mean(diffs))
    return {"ok": m >= PIXEL_DIFF_MIN, "mean_abs_diff": m, "pairs": n,
            "why": f"평균 |화소차| {m:.2f} (기준 >= {PIXEL_DIFF_MIN})"}


def summarise(rows: list[dict]) -> dict:
    """rows: [{group, image, instruction, argmax, margin}] -> 묶음별 요약. 모수 포함."""
    out: dict[str, dict] = {}
    for g in GROUPS:
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            out[g] = {"n": 0}
            continue
        counts: dict[str, int] = {}
        for r in rs:
            counts[r["argmax"]] = counts.get(r["argmax"], 0) + 1
        top = max(counts, key=lambda k: counts[k])
        out[g] = {"n": len(rs),
                  "margin_median": float(st.median(r["margin"] for r in rs)),
                  "argmax_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                  "top": top, "top_share": counts[top] / len(rs)}
    return out


def _dist(a: dict, b: dict) -> float:
    """Total-variation distance between two argmax distributions. 분포 거리 0~1."""
    keys = set(a) | set(b)
    na, nb = sum(a.values()) or 1, sum(b.values()) or 1
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


def judge(summary: dict, blank: dict, tally: dict) -> dict:
    """Decide, refusing to decide when it cannot. 못 재면 못 잰다고 한다."""
    rows = []

    def row(name, ok, got, want):
        rows.append({"name": name, "ok": ok, "got": got, "want": want})

    for g in GROUPS:
        n = summary.get(g, {}).get("n", 0)
        row(f"{g} 장수", n >= MIN_PER_GROUP, f"{n}", f">= {MIN_PER_GROUP}")
    row("빈 장면 검정 (empty != can)", blank.get("ok"), blank.get("why"),
        f"평균 |화소차| >= {PIXEL_DIFF_MIN}")

    e, c = summary.get("empty", {}), summary.get("can", {})
    if e.get("n") and c.get("n"):
        dm = abs(e["margin_median"] - c["margin_median"])
        da = _dist(e["argmax_counts"], c["argmax_counts"])
        row("margin 이 장면에 따라 달라진다", dm >= DEP_MARGIN_MIN,
            f"|{e['margin_median']:.4f} - {c['margin_median']:.4f}| = {dm:.4f}",
            f">= {DEP_MARGIN_MIN}")
        row("argmax 분포가 장면에 따라 달라진다", da >= DEP_ARGMAX_MIN,
            f"{da:.3f}", f">= {DEP_ARGMAX_MIN}")
    else:
        row("margin 이 장면에 따라 달라진다", None, "empty 또는 can 이 비었다", "-")
        row("argmax 분포가 장면에 따라 달라진다", None, "empty 또는 can 이 비었다", "-")

    p = sum(1 for r in rows if r["ok"] is True)
    f = sum(1 for r in rows if r["ok"] is False)
    u = sum(1 for r in rows if r["ok"] is None)
    verdict = "PASS" if f == 0 and u == 0 else ("FAIL" if f else "INCOMPLETE")
    concl = ("이미지가 기여한다 — 거부 임계를 논할 수 있다" if verdict == "PASS" else
             "이미지 의존성이 확인되지 않았다 — 빈 장면 거부는 이 구조로 불가능하다"
             if f else "판정 불가")
    return {"rows": rows, "passed": p, "failed": f, "unknown": u, "total": len(rows),
            "verdict": verdict, "conclusion": concl, "tally": tally,
            "blank_test": blank, "by_group": summary,
            "note": "강제선택이라 '거부' 출력은 존재하지 않는다. 여기서 재는 것은 의존성이다"}


def score_all(groups: dict[str, list[Path]], scorer, instructions: list[str]) -> list[dict]:
    """scorer(image_path, instruction) -> (argmax_name, margin). 주입해서 검증 가능하게 둔다."""
    rows = []
    for g in GROUPS:
        for p in groups[g]:
            for ins in instructions:
                name, margin = scorer(p, ins)
                rows.append({"group": g, "image": p.name, "instruction": ins,
                             "argmax": name, "margin": float(margin)})
    return rows


def real_scorer(model_id: str, device: str, n_instr: int):
    """Build the real VLM scorer. 실제 VLM 채점기. 규칙은 skill_choice 그대로 쓴다."""
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    from vlm.skill_choice import INSTRUCTIONS, SKILLS, forced_choice_scores

    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16).to(device).eval()

    instr = [INSTRUCTIONS[s][i] for i in range(max(1, n_instr // len(SKILLS)))
             for s in SKILLS][:n_instr]

    def scorer(path: Path, instruction: str):
        img = Image.open(path).convert("RGB")
        idx, margin, _ = forced_choice_scores(model, proc, img, instruction, device)
        return SKILLS[idx], margin

    return scorer, instr


# ── 자체검증 (모델·사진 없이 돈다) ───────────────────────────────────────────
def selftest() -> int:
    import tempfile

    import numpy as np
    ok = tot = 0

    def chk(name, cond, note=""):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(f"  {'OK  ' if cond else '실패'} {name}   {note}")

    def fake(rows_spec):
        return [{"group": g, "image": f"{g}_{i}.jpg", "instruction": "x",
                 "argmax": a, "margin": m}
                for g, lst in rows_spec.items() for i, (a, m) in enumerate(lst)]

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for g, k in (("empty", 12), ("can", 12), ("other", 10), ("occluded", 10)):
            for i in range(k):
                (d / f"{g}_{i:02d}.jpg").write_bytes(b"x")
        (d / "노이즈.txt").write_text("x")
        (d / "unknown_1.jpg").write_bytes(b"x")
        groups, tally = collect(d)
        chk("1 접두어로 묶고 모수를 낸다",
            tally["읽은 파일"] == 45 and tally["접두어 불명"] == 1 and len(groups["empty"]) == 12,
            f"{tally}")

    # [2-3] 의존성이 있는 경우 / 없는 경우
    dep = fake({"empty": [("pick_place", 0.10)] * 12, "can": [("sort_two", 0.40)] * 12,
                "other": [("pick_place", 0.2)] * 10, "occluded": [("sort_two", 0.3)] * 10})
    s = summarise(dep)
    j = judge(s, {"ok": True, "why": "평균 |화소차| 9.00"}, {"읽은 파일": 44})
    chk("2 장면에 따라 달라짐 -> PASS", j["verdict"] == "PASS",
        f"{j['verdict']} · 통과 {j['passed']}/{j['total']}")

    same = fake({g: [("pick_place", 0.25)] * 12 for g in GROUPS})
    j2 = judge(summarise(same), {"ok": True, "why": "평균 |화소차| 9.00"}, {})
    chk("3 이미지 무관 -> FAIL (판별행)",
        j2["verdict"] == "FAIL" and "불가능" in j2["conclusion"],
        f"{j2['verdict']} · 불합격 {j2['failed']}")

    # [4] 빈 장면 검정 실패 -> 미판정. '차이 없음'이 공짜로 나오면 안 된다
    j3 = judge(summarise(same), {"ok": False, "why": "평균 |화소차| 0.30"}, {})
    chk("4 empty==can 그림 -> 검정 불합격 (판별행)",
        any(r["name"].startswith("빈 장면") and r["ok"] is False for r in j3["rows"]))

    # [5] 묶음 하나가 비면 판정하지 않는다
    part = fake({"empty": [("a", 0.1)] * 12, "can": [], "other": [], "occluded": []})
    j4 = judge(summarise(part), {"ok": None, "why": "한쪽이 비었다"}, {})
    chk("5 묶음 비면 미판정 (판별행)", j4["verdict"] != "PASS" and j4["unknown"] >= 1,
        f"{j4['verdict']} · 미판정 {j4['unknown']}")

    # [6] 분포 거리 정답 아는 행
    chk("6 분포 거리 정답", abs(_dist({"a": 10}, {"b": 10}) - 1.0) < 1e-9
        and abs(_dist({"a": 10}, {"a": 10})) < 1e-9
        and abs(_dist({"a": 10}, {"a": 5, "b": 5}) - 0.5) < 1e-9, "1.0 · 0.0 · 0.5")

    # [7] 주입한 채점기로 끝까지 흐른다
    g2 = {"empty": [Path("empty_1.jpg")], "can": [Path("can_1.jpg")],
          "other": [], "occluded": []}
    rows = score_all(g2, lambda p, i: ("pick_place", 0.3 if p.name.startswith("can") else 0.1),
                     ["지시문"])
    chk("7 채점기 주입 -> 행 생성", len(rows) == 2 and rows[0]["group"] == "empty",
        f"{len(rows)}행")

    # [8] 장수 미달이 잡힌다
    few = fake({g: [("a", 0.1)] * 3 for g in GROUPS})
    j5 = judge(summarise(few), {"ok": True, "why": ""}, {})
    chk("8 장수 미달 -> 불합격 (판별행)",
        sum(1 for r in j5["rows"] if r["name"].endswith("장수") and r["ok"] is False) == 4,
        f"불합격 {j5['failed']}")

    print(f"\n자체검증 {ok}/{tot}")
    return 0 if ok == tot else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--photos")
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--instructions", type=int, default=5, help="사진당 지시문 수")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if not a.photos:
        ap.error("--photos 가 필요하다 (또는 --selftest)")

    print("계측기 자체검증 먼저 —")
    if selftest() != 0:
        raise SystemExit("!! 자체검증 실패 — 실데이터를 재지 않는다")

    photos = Path(a.photos).expanduser()
    groups, tally = collect(photos)
    print(f"\n사진 묶음 {tally}")
    blank = blank_test(groups)
    print(f"빈 장면 검정 — {blank.get('why')}")
    if blank.get("ok") is not True:
        print("** empty 와 can 이 같은 그림이면 '차이 없음'이 공짜로 나온다. 재지 않는다 **")
        raise SystemExit(2)

    scorer, instr = real_scorer(a.model, a.device, a.instructions)
    print(f"지시문 {len(instr)}개 · 사진 {tally['묶인 파일']}장 "
          f"-> 추론 {len(instr) * tally['묶인 파일']}회")
    rows = score_all(groups, scorer, instr)
    r = judge(summarise(rows), blank, tally)

    print()
    for g in GROUPS:
        s = r["by_group"][g]
        if s["n"]:
            print(f"  {g:<10} n {s['n']:<4} margin 중앙 {s['margin_median']:+.4f} "
                  f"· 최빈 {s['top']} {s['top_share'] * 100:.0f}%")
    print("\n── 사전 등록 게이트 ──")
    for row in r["rows"]:
        mark = {True: " 통과 ", False: "불합격 ", None: "미판정 "}[row["ok"]]
        print(f"  [{mark}] {row['name']:<28} {row['got']}   기준 {row['want']}")
    print(f"\n  통과 {r['passed']} · 불합격 {r['failed']} · 미판정 {r['unknown']} / {r['total']}")
    print(f"  판정: {r['verdict']} — {r['conclusion']}")
    print(f"  ** {r['note']} **")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({**r, "rows_raw": rows}, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
