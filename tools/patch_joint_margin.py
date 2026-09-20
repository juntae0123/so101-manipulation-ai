#!/usr/bin/env python3
"""Add joint-limit margin reporting to check_real_traj_ik.py (idempotent).
check_real_traj_ik.py 에 관절 한계 여유 계측을 더한다 (멱등).

왜 필요한가
-----------
지금 계측기는 웨이포인트마다 "IK 가 풀렸다 / 안 풀렸다" 만 낸다.
한계에서 0.1 도 떨어져 겨우 풀린 편과 40 도 여유로 풀린 편이 **같은 출력**이다.
2026-09-19 현석 실물 로그에서 이게 터졌다 — wrist_flex 가 시작부터 하한(-95.01 도)
15.6 도 앞에 있었고 첫 청크 0.4 초에 여유를 다 썼다. 오프라인 IK 는 그 편을
"통과" 로 찍었다. 통과 여부만으로는 예측이 안 된다.

무엇을 더하나
-------------
편마다 관절별 `min(q - qlo, qhi - q)` 의 최소값을 도 단위로 낸다.
IK 가 한 번도 안 풀린 편은 None 이 아니라 사유를 명시한다 —
"여유를 못 쟀다" 와 "여유가 넉넉하다" 가 같은 출력이 되면 안 된다.

되돌리는 법
-----------
    python patch_joint_margin.py --revert path/to/check_real_traj_ik.py
원본을 .bak_joint_margin 으로 남긴다.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "# [joint_margin_patch]"

ANCHOR_COLLECT = """        if why is None:
            qs.append(q)
            seed = q"""

INSERT_COLLECT = """        if why is None:
            qs.append(q)
            seed = q
            # [joint_margin_patch] 한계까지 남은 여유를 관절별로 적재한다.
            #    통과 여부만으로는 아슬아슬한 편과 여유 있는 편이 구분되지 않는다.
            margins.append(np.minimum(np.asarray(q) - qlo, qhi - np.asarray(q)))"""

# 두 버전을 다 먹는다. 로컬 618줄판과 서버 628줄판(splits 추가)의 선언이 다르다.
INIT_VARIANTS = [
    ("""    qs, reasons, residuals = [], [], []""",
     """    qs, reasons, residuals = [], [], []
    margins: list = []                        # [joint_margin_patch]"""),
    ("""    qs, reasons, residuals, splits = [], [], [], []""",
     """    qs, reasons, residuals, splits = [], [], [], []
    margins: list = []                        # [joint_margin_patch]"""),
]

ANCHOR_RETURN = """        "reasons": reasons,
    }"""

INSERT_RETURN = '''        "reasons": reasons,
        # [joint_margin_patch] 여유 계측. 모수(scored/총)를 항상 같이 낸다.
        **_margin_block(margins, len(ok_way)),
    }


def _margin_block(margins: list, n_waypoints: int) -> dict:
    """Summarize per-joint distance to the nearest joint limit, in degrees.
    관절별 한계까지 남은 여유를 도 단위로 요약한다.

    IK 가 한 번도 안 풀린 편은 여유를 잴 수 없다. 그 경우 값을 None 으로 두되
    사유를 같이 실어서, '여유 없음' 과 '여유 못 쟀음' 이 같은 출력이 되지 않게 한다.
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
    M = np.degrees(np.stack(margins))              # (scored, n_joints)
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
        print(f"\\n관절 여유 — 잰 편 {len(scored)}/{len(results)}"
              f" (못 잰 편 {unscored}: IK 전무)")
        if not scored:
            print("  ⚠️ 한 편도 못 쟀다. 여유가 넉넉한 것이 아니라 계측이 안 된 것이다")
            return
        vals = sorted(r["joint_margin_min_overall_deg"] for r in scored)
        for thr in (1.0, 5.0, 10.0):
            n = sum(1 for v in vals if v < thr)
            print(f"  여유 {thr:4.1f}도 미만  {n}/{len(scored)}편")
        mid = vals[len(vals) // 2]
        print(f"  최소 {vals[0]:.2f}도 · 중앙 {mid:.2f}도 · 최대 {vals[-1]:.2f}도")
        from collections import Counter
        c = Counter(r["tightest_joint_index"] for r in scored)
        print("  가장 빡빡한 관절: " + " · ".join(
            f"j{k} {v}/{len(scored)}편" for k, v in sorted(c.items())))

    def rate(x, n):'''


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

    init_hit = [(a, b) for a, b in INIT_VARIANTS if src.count(a) == 1]
    if len(init_hit) != 1:
        print(f"선언 앵커 후보 {len(INIT_VARIANTS)}개 중 정확히 1회 잡힌 것 {len(init_hit)}개 — 중단")
        for a, _ in INIT_VARIANTS:
            print(f"  발견 {src.count(a)}회  {a.strip()[:60]}")
        return 2
    edits = [
        (*init_hit[0], "margins 초기화"),
        (ANCHOR_COLLECT, INSERT_COLLECT, "여유 적재"),
        (ANCHOR_RETURN, INSERT_RETURN, "반환 블록 + _margin_block"),
        (ANCHOR_SUMMARY, INSERT_SUMMARY, "집계 요약"),
    ]
    missing = [name for anchor, _, name in edits if src.count(anchor) != 1]
    if missing:
        print("앵커를 정확히 1회 찾지 못했다 — 중단한다:")
        for anchor, _, name in edits:
            print(f"  {name:20s} 발견 {src.count(anchor)}회 (1이어야 한다)")
        return 2

    if not bak.exists():
        shutil.copy2(path, bak)
    for anchor, insert, _ in edits:
        src = src.replace(anchor, insert, 1)
    path.write_text(src, encoding="utf-8")
    print(f"적용했다: {path}  (백업 {bak.name})")
    for _, _, name in edits:
        print(f"  ✓ {name}")
    return 0


def selftest() -> int:
    """Self-check with known-answer and discriminating rows.
    정답 아는 행과 판별 행으로 자체 검증한다."""
    import numpy as np  # noqa: F401 — _margin_block 이 쓴다

    ns: dict = {"np": np}
    exec(INSERT_RETURN.split("def _margin_block", 1)[1].join(["def _margin_block", ""]), ns)
    mb = ns["_margin_block"]

    passed = failed = 0

    def check(name: str, got, want) -> None:
        nonlocal passed, failed
        ok = got == want
        passed, failed = passed + ok, failed + (not ok)
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {got!r}" + ("" if ok else f" != {want!r}"))

    # 1) 정답 아는 행 — 여유 0.1 rad 하나뿐이면 최소가 그 값이어야 한다
    r = mb([np.array([0.5, 0.1, 0.9])], 10)
    check("최소 여유 = 5.729도", r["joint_margin_min_overall_deg"], round(float(np.degrees(0.1)), 3))
    check("빡빡한 관절 = 1", r["tightest_joint_index"], 1)
    check("모수 scored=1", r["margin_scored_waypoints"], 1)
    check("모수 total=10", r["margin_total_waypoints"], 10)

    # 2) 판별 행 — 여유 없음(빈 입력)이 '넉넉함'으로 나오면 안 된다
    r0 = mb([], 7)
    check("못 잰 편은 None", r0["joint_margin_min_overall_deg"], None)
    check("못 잰 편은 사유 명시", r0["margin_unmeasured_reason"] is not None, True)
    check("못 잰 편도 분모는 남는다", r0["margin_total_waypoints"], 7)

    # 3) 판별 행 — 여러 웨이포인트 중 최악이 골라져야 한다 (평균이 아니라)
    r2 = mb([np.array([1.0, 1.0]), np.array([1.0, 0.01])], 2)
    check("최악 선택(평균 아님)", r2["joint_margin_min_overall_deg"], round(float(np.degrees(0.01)), 3))

    # 4) 판별 행 — 한계를 넘어선 음수 여유가 음수로 보고돼야 한다
    r3 = mb([np.array([-0.05, 0.4])], 1)
    check("한계 초과는 음수로", r3["joint_margin_min_overall_deg"] < 0, True)

    print(f"\n자체검증 {passed}/{passed + failed}")
    return 0 if failed == 0 else 1


def main() -> None:
    """CLI entry point. 명령행 진입점."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
