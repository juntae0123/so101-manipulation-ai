#!/usr/bin/env python3
"""Catch known instrument anti-patterns before they ship.
알려진 계측기 안티패턴을 배포 전에 잡는다.

왜 있나 — 2026-09-15~16 에 **같은 병의 계측기 결함이 6건** 나왔다. 전부
"없음" 과 "정상" 이 같은 출력을 내는 형태였고, 전부 **사전에 알 수 있는** 것이었다.
6건을 겪은 뒤 TS 문서에 규칙을 적었는데, 그 다음 날 아침 내가 만든 계측기가
그 규칙 첫 줄을 어겼다. **문서만으로는 안 고쳐진다.**

그래서 검사 가능한 형태로 옮긴다. 이 도구는 문법이 아니라 **판정 구조**를 본다.

    # [서버] 또는 # [로컬]
    python tools/lint_instruments.py              tools/ 와 eval/ 전체
    python tools/lint_instruments.py path1 path2  지정 파일만
    python tools/lint_instruments.py --strict     경고도 실패로 취급

⚠️ 이 도구도 계측기다. 오탐이 많으면 아무도 안 쓰게 되고, 그러면 없는 것과 같다.
   그래서 **확실한 것만** 잡는다. 애매한 것은 규칙으로 두고 코드에 넣지 않았다.

검증 기록 — 만든 날 저장소 116개 파일에 돌려 오탐을 걷어냈다 🟢
  · `minmax-without-guard` 가 `max(..., default=None)` 과 `if x else None` 을 잡았다
    -> 억제 규칙 추가
  · `nan-propagating-reduce` 를 넣었다가 **뺐다.** `max(1, int(seconds * rate))` 처럼
    변수명에 rate/loss 가 들어간 줄을 전부 잡았다. nan 전파는 정규식으로 못 잡는다.
    `규칙_계측기_작성.md` 의 문서 규칙으로만 남긴다
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (심각도, 이름, 정규식, 설명, 어떻게 고치나)
SHELL_RULES: list[tuple[str, str, str, str, str]] = [
    (
        "ERROR", "grep-c-in-and-chain",
        r"grep\s+(-[a-zA-Z]*c[a-zA-Z]*\s)[^|\n]*&&",
        "`grep -c` 는 0건이면 종료코드 1 이다. `&&` 사슬이 거기서 끊겨 "
        "뒤 명령이 통째로 안 돈다",
        "진단 명령은 `;` 로 잇는다. `&&` 는 실패 시 뒤를 막아야 할 때만",
    ),
    (
        "ERROR", "zero-or-more-number",
        r"grep\s+-[a-zA-Z]*o[a-zA-Z]*\s+['\"][^'\"]*\[0-9[^\]]*\]\*",
        "`[0-9.]*` 는 **0개 이상**이라 숫자가 없어도 매칭된다. "
        "안내문의 `val ` 같은 것이 빈 값으로 잡힌다",
        "`[0-9]+` 또는 `[0-9]+\\.[0-9]+` 로 1개 이상을 요구한다",
    ),
    (
        "ERROR", "compare-without-empty-check",
        r"\[\s+\"\$[A-Za-z_][A-Za-z0-9_]*\"\s*=\s*\"\$[A-Za-z_][A-Za-z0-9_]*\"\s+\]",
        "명령치환 결과를 빈 검사 없이 비교하면 **빈 값끼리 같다**가 나온다. "
        "실패가 통과로 보인다",
        "비교 전에 `[ -z \"$A\" ] && exit 1` 로 죽인다",
    ),
    (
        "WARN", "no-set-u",
        r"\A(?!.*set -[a-z]*u)",
        "`set -u` 가 없다. 오타난 변수가 빈 문자열로 조용히 퍼진다",
        "`set -u` (또는 `set -eu`) 를 맨 위에 둔다",
    ),
    (
        "WARN", "grep-case-sensitive-status",
        r"grep\s+(?!-[a-zA-Z]*i)[^|\n]*['\"][^'\"]*(skip|invalid|error|fail|warn)"
        r"[^'\"]*['\"]",
        "상태 문자열을 소문자로만 찾는다. 로그가 `SKIP`·`INVALID` 면 못 잡는다",
        "`grep -i` 를 쓰거나 대소문자를 모두 패턴에 넣는다",
    ),
]

PY_RULES: list[tuple[str, str, str, str, str]] = [
    (
        "ERROR", "bare-except-pass",
        r"except[^\n:]*:\s*\n\s*pass\b",
        "예외를 삼킨다. 실패가 정상처럼 보인다",
        "최소한 무엇이 실패했는지 찍는다. 판정 경로면 죽인다",
    ),
    (
        "ERROR", "minmax-without-guard",
        r"(?<![\w.])(?:min|max)\(\s*(?:\[|\()?[a-zA-Z_][\w.]*\s+for\s",
        "빈 제너레이터에 `min`/`max` 를 쓰면 ValueError 로 죽거나, "
        "`default=` 를 주면 그 값이 결과로 둔갑한다",
        "먼저 개수를 세고, 0 이면 '측정 불가' 로 보고한다",
    ),
    (
        "WARN", "float-equality",
        r"(?<![=!<>])==\s*(?:float\(|[0-9]+\.[0-9]+)",
        "부동소수 동등 비교. 재현성 검사에서는 의도일 수 있으나 대개 버그다",
        "허용 오차로 비교하거나, 의도면 주석으로 이유를 남긴다",
    ),
]

# 집합 연산·필터 결과를 찍으면서 모수를 같이 찍지 않는 함수. 줄 단위로는
# 판정할 수 없어 파일 단위 휴리스틱으로 본다.
SET_OP = re.compile(r"print\([^)]*(?:set\(|&\s*set\(|\.intersection\()")
COUNT_HINT = re.compile(r"len\(|찾음|전체|모수|n=%d|/ 전체")


# 한 줄 안에 이것이 있으면 그 줄의 해당 규칙은 이미 막혀 있는 것으로 본다.
# 오탐이 남으면 아무도 이 도구를 안 쓰게 되고, 그러면 없는 것과 같다.
SUPPRESS = {
    # `default=` 를 줬거나 `if ... else` 로 빈 경우를 갈랐으면 이미 막혀 있다.
    "minmax-without-guard": re.compile(r"default\s*=|\bif\b.*\belse\b"),
    "float-equality": re.compile(r"#\s*(?:의도|intentional|재현)"),
}


def scan(path: Path, rules) -> list[tuple[str, int, str, str, str]]:
    """Return (severity, line, rule, message, fix) for each hit in one file.
    파일 하나에서 걸린 것들을 돌려준다."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    hits = []
    for sev, name, pat, msg, fix in rules:
        flags = re.M | re.S if name == "no-set-u" else re.M
        for m in re.finditer(pat, text, flags):
            line = text[: m.start()].count("\n") + 1
            sup = SUPPRESS.get(name)
            if sup is not None and line <= len(lines) and sup.search(lines[line - 1]):
                continue
            hits.append((sev, line, name, msg, fix))
            if name == "no-set-u":
                break
    # 집합 연산 + 모수 없음
    if path.suffix == ".py" and SET_OP.search(text) and not COUNT_HINT.search(text):
        line = text[: SET_OP.search(text).start()].count("\n") + 1
        hits.append((
            "ERROR", line, "set-op-without-denominator",
            "집합 연산 결과를 찍으면서 모수를 안 찍는다. 입력이 비어 있어서 나온 "
            "빈 결과와 '문제 없음' 이 구별되지 않는다",
            "찾은 개수와 전체 개수를 같이 찍는다",
        ))
    return sorted(hits, key=lambda h: h[1])


def main() -> int:
    """Entry point.
    진입점."""
    ap = argparse.ArgumentParser(description="계측기 안티패턴 검사")
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--strict", action="store_true", help="WARN 도 실패로 친다")
    args = ap.parse_args()

    if args.paths:
        files = [p for p in args.paths if p.is_file()]
    else:
        files = sorted(
            p for d in ("tools", "eval", "data", "policy")
            for p in (ROOT / d).rglob("*")
            if p.suffix in (".py", ".sh") and p.name != "lint_instruments.py"
        )
    if not files:
        print("!! 검사할 파일이 없다. 경로를 확인하라")
        return 2

    n_err = n_warn = 0
    for f in files:
        rules = SHELL_RULES if f.suffix == ".sh" else PY_RULES
        for sev, line, name, msg, fix in scan(f, rules):
            mark = "🔴" if sev == "ERROR" else "🟡"
            rel = f.relative_to(ROOT)
            print(f"{mark} {rel}:{line}  [{name}]")
            print(f"   {msg}")
            print(f"   -> {fix}")
            n_err += sev == "ERROR"
            n_warn += sev == "WARN"

    # 모수를 찍는다. 이 도구가 자기 규칙을 어기면 안 된다.
    print(f"\n검사 {len(files)}개 파일 · 🔴 {n_err}건 · 🟡 {n_warn}건")
    if n_err == 0 and n_warn == 0:
        print("걸린 것 없음")
    fail = n_err > 0 or (args.strict and n_warn > 0)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
