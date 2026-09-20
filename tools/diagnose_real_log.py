#!/usr/bin/env python3
"""Read a real-robot run log and say what went wrong, with evidence counts.
실물 실행 로그를 읽고 무엇이 틀렸는지 말한다. 근거 건수를 같이 낸다.

왜 필요한가
-----------
2026-09-19 실물에서 팔이 물체로 가지 않고 반경 565mm 원호만 돌았다. 로그에는
답이 있었지만 찾는 데 오래 걸렸다. **같은 증상을 다시 만나면 몇 초 안에 갈라야 한다.**

설계 원칙
    - 필드를 못 찾으면 **미검출**이다. 정상이 아니다
    - 규칙마다 근거 건수와 검사한 구간 수를 같이 낸다 (모수)
    - 로그 형식을 하드코딩하지 않는다. `이름 = 값` · `이름: 값` · `이름 값` 을 모두 줍는다
      대신 **무엇을 주웠는지 목록으로 찍는다** — 형식이 바뀌면 그게 보인다

Usage
    python diagnose_real_log.py --selftest
    python diagnose_real_log.py --log run.txt --out out/diag.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
KV = re.compile(rf"([A-Za-z_][A-Za-z0-9_]*)\s*[=:]?\s*({NUM})")
BOOL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(True|False|true|false)")
APPROACH_LIMIT = 1.0     # 접근 오차 상한. --approach-limit 로 덮는다
CYCLE = re.compile(r"(?:cycle|청크|chunk|step|사이클)\s*[#=:]?\s*(\d+)", re.I)

# 이 이름들이 로그에 있어야 진단이 된다. 없으면 '미검출' 로 찍고 규칙을 건너뛴다
EXPECT = ("approach_error", "residual", "ik_ok", "limits_ok", "wrist_roll",
          "wrist_flex", "gap", "gripper")


def parse(text: str) -> dict:
    """Split into cycles and collect named numbers. 사이클로 나누고 이름 붙은 수를 줍는다."""
    cycles: list[dict] = []
    cur: dict = {"index": None, "fields": {}, "lines": 0}
    names: set[str] = set()
    bools: dict[str, list[bool]] = {}
    for line in text.splitlines():
        m = CYCLE.search(line)
        if m:
            if cur["lines"]:
                cycles.append(cur)
            cur = {"index": int(m.group(1)), "fields": {}, "lines": 0}
        cur["lines"] += 1
        for k, v in KV.findall(line):
            names.add(k)
            cur["fields"].setdefault(k, []).append(float(v))
        for k, v in BOOL.findall(line):
            names.add(k)
            bools.setdefault(k, []).append(v.lower() == "true")
            cur["fields"].setdefault(k + "__bool", []).append(1.0 if v.lower() == "true" else 0.0)
    if cur["lines"]:
        cycles.append(cur)
    found = sorted(n for n in names if any(e in n.lower() for e in EXPECT))
    return {"cycles": cycles, "names": sorted(names), "relevant": found,
            "relevant_count": len(found), "expected_count": len(EXPECT)}


def _series(cycles: list[dict], key: str) -> list[list[float]]:
    """Per-cycle series for a field, matched case-insensitively by substring."""
    out = []
    for c in cycles:
        vals: list[float] = []
        for k, v in c["fields"].items():
            if key.lower() in k.lower():
                vals.extend(v)
        if vals:
            out.append(vals)
    return out


def _monotone_rising(xs: list[float]) -> bool:
    return len(xs) >= 3 and all(b > a for a, b in zip(xs, xs[1:]))


def rules(p: dict) -> list[dict]:
    """Symptom rules. 증상 규칙. 근거 건수와 모수를 항상 함께 낸다."""
    cy = p["cycles"]
    out: list[dict] = []

    def add(rid, name, hit, n, note, fix, missing=False):
        out.append({"id": rid, "name": name, "hits": hit, "checked": n,
                    "verdict": "미검출" if missing else ("해당" if hit else "해당 없음"),
                    "note": note, "fix": fix})

    # R1 청크 안에서 오차가 단조 증가하고 청크마다 리셋된다 -> 앵커/누적 규약
    ae = _series(cy, "approach_error")
    if not ae:
        add("R1", "청크 내 오차 누적", 0, 0, "approach_error 를 못 찾았다", "로그에 그 값을 찍게 해라", missing=True)
    else:
        hit = sum(1 for s in ae if _monotone_rising(s))
        add("R1", "청크 내 오차 누적", hit, len(ae),
            f"청크 안에서 단조 증가한 청크 {hit} / 검사 {len(ae)}",
            "so101_infer.unroll 이 청크 안에서 누적하는지 본다. 앵커(T0 @ A[k]) 여야 한다 (D-AI-80)")

    # R2 IK 잔차는 작은데 접근 오차가 크다 -> IK 가 아니라 목표가 틀렸다
    res = [v for s in _series(cy, "residual") for v in s]
    aef = [v for s in ae for v in s]
    if not res or not aef:
        add("R2", "목표 좌표 오류", 0, 0, "residual 또는 approach_error 미검출", "둘 다 로그에 찍어라", missing=True)
    else:
        # ⚠️ approach_error 의 단위는 로그마다 다를 수 있다. 기본 1.0 은
        #    2026-09-19 실물 로그(정상 0.11 수준 / 고장 3.57까지 증가)에서 잡은 값이다.
        #    다른 로그를 볼 때는 --approach-limit 로 바꾸고 conditions 에 적어라.
        lim = APPROACH_LIMIT
        small = sum(1 for v in res if v < 1.0)
        big = sum(1 for v in aef if v > lim)
        hit = 1 if (small / len(res) > 0.9 and big / len(aef) > 0.3) else 0
        add("R2", "목표 좌표 오류", hit, len(res),
            f"IK 잔차 1mm 미만 {small}/{len(res)} · 접근 오차 {lim} 초과 {big}/{len(aef)}",
            "IK 는 정상이고 목표가 틀린 것이다. 좌표계·앵커·프레임 정렬을 본다")

    # R3 손목 관절이 한계에 붙어 있다 -> 자세 도달 불가 (5축 한계)
    wf = [v for s in _series(cy, "wrist_flex") for v in s]
    if not wf:
        add("R3", "손목 한계 포화", 0, 0, "wrist_flex 미검출", "관절값을 로그에 찍어라", missing=True)
    else:
        lim = 1.658
        hit = sum(1 for v in wf if abs(abs(v) - lim) < 0.02)
        add("R3", "손목 한계 포화", hit, len(wf),
            f"wrist_flex 가 한계 {lim} 에 붙은 표본 {hit} / {len(wf)}",
            "5축으로 만족 못 하는 자세다. 측면 파지로 바꾸거나 물체 위치를 옮긴다")

    # R4 IK 해가 계속 없다 -> 도달 포락선 밖
    ik = [v for s in _series(cy, "ik_ok__bool") for v in s]
    if not ik:
        add("R4", "도달 범위 밖", 0, 0, "ik_ok 미검출", "IK 성공 여부를 로그에 찍어라", missing=True)
    else:
        fail = sum(1 for v in ik if v == 0.0)
        add("R4", "도달 범위 밖", 1 if fail / len(ik) > 0.3 else 0, len(ik),
            f"IK 실패 {fail} / {len(ik)}",
            "측면 파지 포락선(z 0.027 -> x 0.37~0.46)을 벗어났다. 물체를 범위 안으로")

    # R5 wrist_roll 이 급변한다 -> IK 해 분기 점프
    wr = _series(cy, "wrist_roll")
    flat = [v for s in wr for v in s]
    if len(flat) < 2:
        add("R5", "IK 해 분기 점프", 0, 0, "wrist_roll 미검출", "관절값을 로그에 찍어라", missing=True)
    else:
        jumps = sum(1 for a, b in zip(flat, flat[1:]) if abs(b - a) > 1.0)
        add("R5", "IK 해 분기 점프", jumps, len(flat) - 1,
            f"1.0 rad 넘는 급변 {jumps} / {len(flat) - 1}",
            "IK 가 다른 해로 건너뛰었다. 직전 해를 시드로 이어 풀고 한계 여유를 검사한다")

    # R6 전송이 0회다 -> 안전 검사에 걸렸다 (이건 정상 동작이다)
    sent = [v for s in _series(cy, "sent") for v in s] or [v for s in _series(cy, "전송") for v in s]
    add("R6", "안전 검사로 전송 차단", 1 if (sent and sum(sent) == 0) else 0, len(sent),
        f"전송 합계 {sum(sent) if sent else '미검출'} / 기록 {len(sent)}건",
        "프리플라이트 불합격이다. **고장이 아니라 설계대로다.** 불합격 사유를 보고 안전 설정을 채운다",
        missing=not sent)

    # R7 그리퍼 폭을 못 읽었다
    gp = [v for s in _series(cy, "gap") for v in s] + [v for s in _series(cy, "gripper") for v in s]
    add("R7", "그리퍼 폭 결측", 0 if gp else 1, len(gp),
        f"개구 표본 {len(gp)}건",
        "폭을 못 읽으면 값을 지어내지 말고 멈춰야 한다 (R5 정정, 2026-09-20)",
        missing=False)
    return out


def summarize(p: dict, rs: list[dict]) -> dict:
    """Rank and state the limits. 순위를 매기고 한계를 적는다."""
    hit = [r for r in rs if r["verdict"] == "해당"]
    miss = [r for r in rs if r["verdict"] == "미검출"]
    return {
        "cycles": len(p["cycles"]),
        "fields_relevant": p["relevant_count"], "fields_expected": p["expected_count"],
        "rules_total": len(rs), "rules_hit": len(hit), "rules_missing": len(miss),
        "top": [r["id"] for r in hit],
        "verdict": ("판정 불가 — 관련 필드가 로그에 없다" if p["relevant_count"] == 0
                    else ("원인 후보 " + ", ".join(r["id"] for r in hit) if hit
                          else "규칙에 걸리는 증상 없음 — 이 규칙집 밖의 원인이다")),
    }


SYNTH_OK = """cycle 0
  residual 0.0021 mm  approach_error 0.11  ik_ok: True  limits_ok: True  wrist_roll 0.21 wrist_flex 0.4 gap 0.070 sent 4
  residual 0.0019 mm  approach_error 0.10  ik_ok: True  limits_ok: True  wrist_roll 0.22 wrist_flex 0.4 gap 0.068 sent 4
cycle 1
  residual 0.0030 mm  approach_error 0.12  ik_ok: True  limits_ok: True  wrist_roll 0.23 wrist_flex 0.4 gap 0.050 sent 4
  residual 0.0028 mm  approach_error 0.11  ik_ok: True  limits_ok: True  wrist_roll 0.23 wrist_flex 0.4 gap 0.040 sent 4
"""

SYNTH_ACC = """cycle 0
  residual 0.0007 mm approach_error 0.11 ik_ok: True wrist_roll 0.2 wrist_flex 0.4 gap 0.07 sent 4
  residual 0.0009 mm approach_error 0.80 ik_ok: True wrist_roll 0.2 wrist_flex 0.4 gap 0.07 sent 4
  residual 0.0011 mm approach_error 1.61 ik_ok: True wrist_roll 0.2 wrist_flex 0.4 gap 0.07 sent 4
  residual 0.0014 mm approach_error 2.53 ik_ok: True wrist_roll 0.2 wrist_flex 0.4 gap 0.07 sent 4
cycle 1
  residual 0.0008 mm approach_error 0.55 ik_ok: True wrist_roll 0.2 wrist_flex 0.4 gap 0.07 sent 4
  residual 0.0010 mm approach_error 1.16 ik_ok: True wrist_roll 0.2 wrist_flex 0.4 gap 0.07 sent 4
  residual 0.0013 mm approach_error 2.22 ik_ok: True wrist_roll 0.2 wrist_flex 0.4 gap 0.07 sent 4
  residual 0.0016 mm approach_error 3.57 ik_ok: True wrist_roll 0.2 wrist_flex 0.4 gap 0.07 sent 4
"""

SYNTH_LIMIT = """cycle 0
  residual 0.5 mm approach_error 0.2 ik_ok: False wrist_flex 1.658 wrist_roll 0.3 gap 0.07 sent 0
  residual 0.6 mm approach_error 0.3 ik_ok: False wrist_flex -1.658 wrist_roll 2.9 gap 0.07 sent 0
"""


def selftest() -> int:
    """Known-answer logs for each rule. 규칙마다 정답 아는 로그를 둔다."""
    log, bad = [], 0

    def chk(n, c, note=""):
        nonlocal bad
        log.append((n, bool(c), note))
        if not c:
            bad += 1

    ok = summarize(parse(SYNTH_OK), rules(parse(SYNTH_OK)))
    chk("1 정상 로그 -> 해당 규칙 없음", ok["rules_hit"] == 0,
        f"해당 {ok['rules_hit']} · 미검출 {ok['rules_missing']} / 규칙 {ok['rules_total']}")

    p = parse(SYNTH_ACC); r = rules(p); s = summarize(p, r)
    chk("2 누적 로그 -> R1 잡는다 (정답 아는 행)", "R1" in s["top"], f"{s['top']}")
    chk("3 누적 로그 -> R2 도 잡는다", "R2" in s["top"], "IK 잔차는 작은데 접근 오차가 크다")

    p2 = parse(SYNTH_LIMIT); s2 = summarize(p2, rules(p2))
    chk("4 한계 포화 -> R3·R4 잡는다 (판별행)",
        "R3" in s2["top"] and "R4" in s2["top"], f"{s2['top']}")
    chk("5 전송 0회 -> R6 잡는다", "R6" in s2["top"], "안전 차단은 고장이 아니라 설계대로다")

    p3 = parse("아무 의미 없는 줄\n또 다른 줄\n")
    s3 = summarize(p3, rules(p3))
    chk("6 빈 로그 -> 판정 불가 (통과 아님)", "판정 불가" in s3["verdict"],
        f"관련 필드 {s3['fields_relevant']}/{s3['fields_expected']}")
    chk("7 미검출이 '해당 없음'과 구분된다 (판별행)", s3["rules_missing"] >= 5,
        f"미검출 {s3['rules_missing']} / {s3['rules_total']}")

    chk("8 사이클 분리", len(parse(SYNTH_ACC)["cycles"]) == 2,
        f"{len(parse(SYNTH_ACC)['cycles'])}개")
    chk("9 모수 보고", ok["fields_relevant"] > 0 and ok["cycles"] == 2,
        f"필드 {ok['fields_relevant']}/{ok['fields_expected']} · 사이클 {ok['cycles']}")

    for nm, o, note in log:
        print(f"  {'OK ' if o else 'FAIL'}  {nm}" + (f"   {note}" if note else ""))
    print(f"\n자체검증 {len(log) - bad}/{len(log)}")
    return 1 if bad else 0


def main() -> None:
    """CLI entry point. 명령행 진입점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--log")
    ap.add_argument("--approach-limit", type=float, default=APPROACH_LIMIT,
                    help="R2 접근 오차 상한. 로그 단위를 확인하고 정해라")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    globals()["APPROACH_LIMIT"] = a.approach_limit
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 판정을 내지 않는다")
    if not a.log:
        ap.error("--log 가 필요하다")

    text = Path(a.log).expanduser().read_text(encoding="utf-8", errors="replace")
    p = parse(text)
    rs = rules(p)
    s = summarize(p, rs)

    print(f"\n사이클 {s['cycles']} · 관련 필드 {s['fields_relevant']} / 기대 {s['fields_expected']}")
    print(f"주운 이름: {', '.join(p['relevant']) or '없음'}\n")
    for r in rs:
        mark = {"해당": "!!", "해당 없음": "  ", "미검출": "??"}[r["verdict"]]
        print(f"{mark} [{r['id']}] {r['name']:<18} {r['verdict']:<6} {r['note']}")
        if r["verdict"] == "해당":
            print(f"        -> {r['fix']}")
    print(f"\n판정: {s['verdict']}")
    print("한계: 이 규칙집에 없는 원인은 못 잡는다. '해당 없음'은 '정상'이 아니다.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"log": str(a.log), "summary": s, "rules": rs, "names": p["names"]},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
