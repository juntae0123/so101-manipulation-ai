#!/usr/bin/env python3
"""Add joint-limit margin reporting to check_real_traj_ik.py (idempotent).
check_real_traj_ik.py 에 관절 한계 여유 계측을 더한다 (멱등).

왜 필요한가
-----------
계측기가 웨이포인트마다 "IK 가 풀렸다 / 안 풀렸다" 만 낸다. 한계에서 0.1도 떨어져
겨우 풀린 편과 40도 여유로 풀린 편이 **같은 출력**이다. 2026-09-19 현석 실물 로그에서
이게 터졌다 — wrist_flex 가 시작부터 하한(-95.01도) 15.6도 앞이었고 첫 청크 0.4초에
여유를 다 썼는데, 오프라인 IK 는 그 편을 "통과" 로 찍었다.

초판 결함 (2026-09-20, 황도경 지적) — 고친 내용
-----------------------------------------------
1. 한계 변수명을 `qlo/qhi` 로 **하드코딩**했는데 대상 파일은 `q_lo/q_hi` 였다.
   성공 IK 가 나오는 순간 NameError. 표본이 0/27 이라 성공 경로를 한 번도 안 타서
   드러나지 않았다. → 이제 **파일에서 실제 이름을 읽어 쓴다.**
2. `margin_summary()` 를 정의만 하고 **호출하지 않았다.** → 호출까지 넣는다.
3. 자체검증이 `_margin_block` 단독 검사라 위 둘을 못 잡았다.
   → **패치를 실제로 적용해 실행까지 하는 통합 판별행**을 넣는다.

되돌리는 법
-----------
    python patch_joint_margin.py --revert path/to/check_real_traj_ik.py
원본을 .bak_joint_margin 으로 남긴다.
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from pathlib import Path

MARKER = "# [joint_margin_patch]"
LIMIT_DECL = re.compile(r"^[ \t]*(\w+),\s*(\w+)\s*=\s*env\.limits\b", re.M)

ANCHOR_COLLECT = """        if why is None:
            qs.append(q)
            seed = q"""

ANCHOR_INIT_RE = re.compile(r"^([ \t]*)qs, reasons, residuals(.*?)= (.*)$", re.M)

ANCHOR_RETURN = '''        "reasons": reasons,
    }'''

INSERT_RETURN = '''        "reasons": reasons,
        # [joint_margin_patch] 여유 계측. 모수(scored/총)를 항상 같이 낸다.
        **_margin_block(margins, len(ok_way)),
    }


def _margin_block(margins: list, n_waypoints: int) -> dict:
    """Summarize per-joint distance to the nearest joint limit, in degrees.
    관절별 한계까지 남은 여유를 도 단위로 요약한다.

    IK 가 한 번도 안 풀린 편은 여유를 잴 수 없다. 값은 None 으로 두되 사유를 같이 실어서
    '여유 없음' 과 '여유 못 쟀음' 이 같은 출력이 되지 않게 한다.
    """
    scored = len(margins)
    if scored == 0:
        return {
            "margin_scored_waypoints": 0,
            "margin_total_waypoints": int(n_waypoints),
            "joint_margin_min_deg": None,
            "joint_margin_min_overall_deg": None,
            "tightest_joint_index": None,
            "margin_unmeasured_reason": "no_successful_ik — 여유를 못 쟀다. 여유가 넉넉한 것이 아니다",
        }
    M = np.degrees(np.stack(margins))
    per_joint = M.min(axis=0)
    j = int(np.argmin(per_joint))
    return {
        "margin_scored_waypoints": int(scored),
        "margin_total_waypoints": int(n_waypoints),
        "joint_margin_min_deg": [round(float(v), 3) for v in per_joint],
        "joint_margin_min_overall_deg": round(float(per_joint[j]), 3),
        "tightest_joint_index": j,
        "margin_unmeasured_reason": None,
    }'''

ANCHOR_SUMMARY = """    def rate(x, n):"""
INSERT_SUMMARY = '''    # [joint_margin_patch] 편 단위 여유 분포. 분모를 항상 같이 찍는다.
    def margin_summary(results: list) -> None:
        scored = [r for r in results if r.get("joint_margin_min_overall_deg") is not None]
        unscored = len(results) - len(scored)
        print(f"\\n관절 여유 — 잰 편 {len(scored)}/{len(results)} (못 잰 편 {unscored}: IK 전무)")
        if not scored:
            print("  ⚠️ 한 편도 못 쟀다. 여유가 넉넉한 것이 아니라 계측이 안 된 것이다")
            return
        vals = sorted(r["joint_margin_min_overall_deg"] for r in scored)
        for thr in (1.0, 5.0, 10.0):
            print(f"  여유 {thr:4.1f}도 미만  {sum(1 for v in vals if v < thr)}/{len(scored)}편")
        print(f"  최소 {vals[0]:.2f}도 · 중앙 {vals[len(vals) // 2]:.2f}도 · 최대 {vals[-1]:.2f}도")
        from collections import Counter
        c = Counter(r["tightest_joint_index"] for r in scored)
        print("  가장 빡빡한 관절: " + " · ".join(
            f"j{k} {v}/{len(scored)}편" for k, v in sorted(c.items())))

    def rate(x, n):'''

ANCHOR_CALL = '''    print("거부 사유:", reason_counts if reason_counts else "없음")'''
INSERT_CALL = '''    print("거부 사유:", reason_counts if reason_counts else "없음")
    margin_summary(results)                       # [joint_margin_patch]'''


def detect_limit_names(src: str) -> tuple[str, str] | None:
    """Read the joint-limit variable names from the file itself. Never hardcode.
    한계 변수명을 파일에서 직접 읽는다. 하드코딩하지 않는다 (초판이 여기서 틀렸다)."""
    hits = LIMIT_DECL.findall(src)
    uniq = set(hits)
    return hits[0] if len(uniq) == 1 else None


def build_edits(src: str) -> tuple[list, str] | tuple[None, str]:
    """Compose the four edits against THIS file's actual names.
    이 파일의 실제 이름에 맞춰 편집 4건을 구성한다."""
    names = detect_limit_names(src)
    if names is None:
        n = len(LIMIT_DECL.findall(src))
        return None, f"한계 변수 선언을 정확히 1종 못 찾았다 (발견 {n}건). 이름을 추측하지 않는다"
    lo, hi = names

    m = ANCHOR_INIT_RE.search(src)
    if not m or len(ANCHOR_INIT_RE.findall(src)) != 1:
        return None, f"qs/reasons/residuals 선언을 정확히 1회 못 찾았다 (발견 {len(ANCHOR_INIT_RE.findall(src))}건)"
    indent = m.group(1)
    init_anchor = m.group(0)
    init_insert = f"{init_anchor}\n{indent}margins: list = []                        {MARKER}"

    collect_insert = (
        f"{ANCHOR_COLLECT}\n"
        f"            {MARKER} 한계까지 남은 여유를 관절별로 적재한다.\n"
        f"            #    통과 여부만으로는 아슬아슬한 편과 여유 있는 편이 구분되지 않는다.\n"
        f"            margins.append(np.minimum(np.asarray(q) - {lo}, {hi} - np.asarray(q)))"
    )

    edits = [
        (init_anchor, init_insert, "margins 초기화"),
        (ANCHOR_COLLECT, collect_insert, f"여유 적재 (한계 변수 {lo}/{hi})"),
        (ANCHOR_RETURN, INSERT_RETURN, "반환 블록 + _margin_block"),
        (ANCHOR_SUMMARY, INSERT_SUMMARY, "집계 함수 정의"),
        (ANCHOR_CALL, INSERT_CALL, "집계 호출"),
    ]
    return edits, f"{lo}/{hi}"


def postcheck(src: str, lo_hi: str) -> list[str]:
    """Verify the patched source, not just that text was inserted.
    삽입 여부가 아니라 **패치된 결과**를 검산한다. 초판은 이 단계가 없었다."""
    fails = []
    try:
        ast.parse(src)
    except SyntaxError as exc:
        fails.append(f"AST 실패: {exc}")
        return fails
    lo, hi = lo_hi.split("/")
    ins = [l for l in src.splitlines() if "margins.append" in l]
    if len(ins) != 1:
        fails.append(f"margins.append 가 {len(ins)}줄 (1 이어야 한다)")
    elif f"- {lo}," not in ins[0] or f"{hi} -" not in ins[0]:
        fails.append(f"여유 계산이 선언된 한계 이름({lo}/{hi})을 안 쓴다: {ins[0].strip()}")
    if src.count("margin_summary(results)") != 1:
        fails.append(f"margin_summary(results) 호출이 {src.count('margin_summary(results)')}회 (1 이어야 한다)")
    if src.count("def margin_summary") != 1:
        fails.append("margin_summary 정의가 1회가 아니다")
    return fails


def apply(path: Path, revert: bool) -> int:
    """Apply or revert the patch. 패치를 적용하거나 되돌린다."""
    bak = path.with_suffix(path.suffix + ".bak_joint_margin")
    if revert:
        if not bak.exists():
            print(f"되돌릴 백업이 없다: {bak}")
            return 1
        shutil.copy2(bak, path)
        print(f"되돌렸다: {path}")
        return 0

    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"이미 적용돼 있다 (멱등): {path}")
        return 0

    edits, info = build_edits(src)
    if edits is None:
        print(f"중단 — {info}")
        return 2
    bad = [(name, src.count(a)) for a, _, name in edits if src.count(a) != 1]
    if bad:
        print("앵커를 정확히 1회 찾지 못했다 — 중단한다:")
        for name, n in bad:
            print(f"  {name:28s} 발견 {n}회 (1이어야 한다)")
        return 2

    out = src
    for anchor, insert, _ in edits:
        out = out.replace(anchor, insert, 1)
    fails = postcheck(out, info)
    if fails:
        print("적용 후 검산 실패 — 파일을 건드리지 않는다:")
        for f in fails:
            print(f"  {f}")
        return 3

    if not bak.exists():
        shutil.copy2(path, bak)
    path.write_text(out, encoding="utf-8")
    print(f"적용했다: {path}  (백업 {bak.name})")
    for _, _, name in edits:
        print(f"  ✓ {name}")
    print("  ✓ 적용 후 검산 통과 (AST · 한계 이름 일치 · 집계 호출 1회)")
    return 0


MOCK = '''import numpy as np


def check_episode(env):
    Q_LOW, Q_HIGH = env.limits[:, 0], env.limits[:, 1]
    qs, reasons, residuals = [], [], []
    for idx in range(3):
        q = np.array([0.1 * idx, 0.9])
        why = None
        if why is None:
            qs.append(q)
            seed = q
        else:
            qs.append(None)
        reasons.append(why)
    ok_way = [r is None for r in reasons]
    return {
        "waypoints_ok": int(sum(ok_way)),
        "reasons": reasons,
    }


def main():
    results = [{"joint_margin_min_overall_deg": 2.0, "tightest_joint_index": 1},
               {"joint_margin_min_overall_deg": None, "tightest_joint_index": None}]
    def rate(x, n):
        return 0
    reason_counts = {}
    print("거부 사유:", reason_counts if reason_counts else "없음")
'''


def selftest() -> int:
    """Unit rows + an INTEGRATION row that actually runs the patched code.
    단위 검사 + **패치된 코드를 실제로 실행하는 통합 판별행**."""
    import tempfile
    import io
    import contextlib
    import numpy as np

    passed = failed = 0

    def check(name, got, want):
        nonlocal passed, failed
        ok = got == want
        passed, failed = passed + ok, failed + (not ok)
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {got!r}" + ("" if ok else f" != {want!r}"))

    ns = {"np": np}
    exec("def _margin_block" + INSERT_RETURN.split("def _margin_block", 1)[1], ns)
    mb = ns["_margin_block"]

    print("— 단위 —")
    r = mb([np.array([0.5, 0.1, 0.9])], 10)
    check("최소 여유 = 5.729도", r["joint_margin_min_overall_deg"], round(float(np.degrees(0.1)), 3))
    check("빡빡한 관절 = 1", r["tightest_joint_index"], 1)
    check("모수 scored=1", r["margin_scored_waypoints"], 1)
    r0 = mb([], 7)
    check("못 잰 편은 None", r0["joint_margin_min_overall_deg"], None)
    check("못 잰 편은 사유 명시", r0["margin_unmeasured_reason"] is not None, True)
    check("못 잰 편도 분모 유지", r0["margin_total_waypoints"], 7)
    r2 = mb([np.array([1.0, 1.0]), np.array([1.0, 0.01])], 2)
    check("최악 선택(평균 아님)", r2["joint_margin_min_overall_deg"], round(float(np.degrees(0.01)), 3))
    check("한계 초과는 음수", mb([np.array([-0.05, 0.4])], 1)["joint_margin_min_overall_deg"] < 0, True)

    print("— 통합 (패치를 실제로 걸고 실행한다) —")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "mock.py"
        f.write_text(MOCK, encoding="utf-8")
        rc = apply(f, False)
        check("모의 파일 패치 성공", rc, 0)
        check("한계 이름을 파일에서 읽었다 (Q_LOW/Q_HIGH)",
              "- Q_LOW," in f.read_text(encoding="utf-8"), True)
        g: dict = {}
        try:
            exec(compile(f.read_text(encoding="utf-8"), str(f), "exec"), g)

            class E:
                limits = np.array([[-1.0, 1.0], [-1.0, 1.0]])

            out = g["check_episode"](E())
            check("성공 IK 경로가 NameError 없이 돈다", out["margin_scored_waypoints"], 3)
            check("여유가 실제로 계산됐다", out["joint_margin_min_overall_deg"] is not None, True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                g["main"]()
            check("집계가 호출된다", "관절 여유 — 잰 편 1/2" in buf.getvalue(), True)
        except Exception as exc:  # noqa: BLE001
            check(f"통합 실행 ({type(exc).__name__}: {exc})", False, True)

        rc2 = apply(f, False)
        check("멱등 (2회차)", rc2, 0)

    print(f"\n자체검증 {passed}/{passed + failed}")
    return 0 if failed == 0 else 1


def main() -> None:
    """CLI entry point. 명령행 진입점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="check_real_traj_ik.py 경로")
    ap.add_argument("--revert", action="store_true", help="백업으로 되돌린다")
    ap.add_argument("--selftest", action="store_true", help="자체 검증만 돌린다")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.target:
        ap.error("target 경로가 필요하다 (또는 --selftest)")
    sys.exit(apply(Path(a.target).expanduser(), a.revert))


if __name__ == "__main__":
    main()
