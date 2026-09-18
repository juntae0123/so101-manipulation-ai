"""Idempotent patch: make simulation/evaluate.py record joint angles per step.
멱등 패치 — evaluate.py 가 스텝마다 관절각(qpos)을 남기게 한다.

왜 필요한가 (2026-09-18)
------------------------
HW 인계 패키지(`handoff/validation/KNOWN_LIMITATIONS.md`)에서 폰 홀더 장착 상태의
기구 간섭이 확인됐다.

    wrist_roll URDF limit  [-2.7438, 2.8412] rad = [-157.2도, +162.8도]
    v010 브래킷 교차        +65도 ~ +162도
    v009 폰 홀더 교차       +90~+105도, +124~+142도

문서 원문: "URDF의 기존 관절 limit을 충돌 없는 작동 범위로 해석하지 않는다."

**우리 시뮬은 이 제약을 모른다.** E1 97.0% · E3 96.0% 는 전 범위를 자유롭게 쓴
결과다. 정책이 wrist_roll 을 +65도 넘게 쓰면 그 롤아웃은 실물에서 브래킷과 충돌한다.

그런데 `evaluate.py` 가 남기는 건 `tcp` 4x4 뿐이라 **어느 관절을 얼마나 썼는지 알 수 없다.**
TCP 에서 역산하면 5축 손목 뒤집기 해가 여럿이고 하필 그 모호성이 wrist_roll 에 걸린다.
역산값은 정책이 쓴 각도가 아니라 **내가 푼 각도**다. 그건 계측이 아니다.

그래서 기록 자체를 고친다. 이 패치 뒤 모든 평가가 관절각을 남긴다.

무엇을 바꾸나
-------------
1) `trajectory_*.json` 각 스텝에 `qpos` (전체 관절 상태) 추가
2) `evaluation.json` 의 각 report 에 `joint_names` / `jnt_qposadr` 추가
   → qpos 인덱스를 이름으로 되짚을 수 있다. 인덱스를 하드코딩하지 않기 위함이다

멱등성
------
삽입 후에만 존재하는 표지 문자열로 판정한다. 2회 이상 돌려도 중복 삽입되지 않는다.
(2026-09-16 에 `patch_train_seed.py` 가 old 가 new 의 부분문자열이라 `--seed` 를
두 번 등록한 사고가 있었다. 같은 실수를 막는다.)

Usage
-----
  python patch_eval_qpos.py --selftest
  python patch_eval_qpos.py --file ~/handoff/simulation/evaluate.py --check
  python patch_eval_qpos.py --file ~/handoff/simulation/evaluate.py
"""
from __future__ import annotations

import argparse
import ast
import shutil
import sys
import tempfile
from pathlib import Path

# ── 패치 규칙 ────────────────────────────────────────────────────────────
# anchor  : 원본에 정확히 1회 나와야 하는 문자열
# insert  : anchor 바로 뒤에 끼워 넣을 문자열
# marker  : 패치 후에만 존재하는 표지. 멱등 판정에 쓴다 (insert 의 부분집합)

RULES: list[dict[str, str]] = [
    {
        "name": "스텝별 qpos 기록",
        "anchor": "trajectories.append({'time':float(env.data.time),",
        "insert": "'qpos':env.data.qpos.tolist(),",
        "marker": "'qpos':env.data.qpos.tolist(),",
    },
    {
        "name": "report 에 관절 이름·주소 기록",
        "anchor": "report={'seed':seed,",
        "insert": ("'joint_names':[env.model.joint(i).name for i in range(env.model.njnt)],"
                   "'jnt_qposadr':[int(a) for a in env.model.jnt_qposadr],"),
        "marker": "'joint_names':[env.model.joint(i).name for i in range(env.model.njnt)],",
    },
]


def inspect(text: str) -> list[dict[str, object]]:
    """Report, per rule, how many anchors and markers the text holds.
    규칙별로 anchor 와 marker 가 몇 개인지 센다. 모수를 같이 낸다."""
    out = []
    for r in RULES:
        out.append({"name": r["name"],
                    "anchors": text.count(r["anchor"]),
                    "markers": text.count(r["marker"])})
    return out


def apply(text: str) -> tuple[str, list[str]]:
    """Apply every rule that is not already applied. 아직 적용 안 된 규칙만 적용한다."""
    notes: list[str] = []
    for r in RULES:
        if r["marker"] in text:
            notes.append(f"건너뜀 (이미 적용됨): {r['name']}")
            continue
        n = text.count(r["anchor"])
        if n != 1:
            raise SystemExit(f"!! anchor 가 {n}번 나온다 (1이어야 한다): {r['name']}\n"
                             f"   anchor = {r['anchor']!r}")
        text = text.replace(r["anchor"], r["anchor"] + r["insert"], 1)
        notes.append(f"적용: {r['name']}")
    return text, notes


# ── 자체 검증 — 정답을 아는 입력만 쓴다 ──────────────────────────────────

STUB = '''import json
def run(env, target, seed, max_lift, max_hold, checkpoint, payload, grip_preload, converted):
    trajectories = []
    trajectories.append({'time':float(env.data.time),'target':target.tolist(),'tcp':env.tcp().tolist(),'width':float(sum(env.data.qpos[env.fids])),
        'object_position_for_scoring':env.data.body('object').xpos.tolist(),
        'contacts_for_scoring':env.contacts(),'continuous_hold_seconds':0.})
    report={'seed':seed,'success':bool(max_hold>=0.5),'max_lift_m':max_lift,'task':env.task}
    return trajectories, report
'''


def selftest() -> int:
    bad = 0

    # [1] 원본에 anchor 가 정확히 1개씩 있나
    inf = inspect(STUB)
    ok = all(i["anchors"] == 1 and i["markers"] == 0 for i in inf)
    for i in inf:
        print(f"[1] {i['name']:24s} anchor {i['anchors']} / marker {i['markers']}")
    print("    ", "OK" if ok else "!! 실패"); bad += (not ok)

    # [2] 1회 적용 후 marker 가 정확히 1개씩
    once, notes = apply(STUB)
    inf1 = inspect(once)
    ok = all(i["markers"] == 1 for i in inf1)
    print(f"[2] 1회 적용 → marker {[i['markers'] for i in inf1]}  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    # [3] 2회·3회 적용해도 marker 가 늘지 않는다 (멱등)
    twice, _ = apply(once)
    thrice, _ = apply(twice)
    inf3 = inspect(thrice)
    ok = all(i["markers"] == 1 for i in inf3) and twice == once == thrice
    print(f"[3] 3회 적용 → marker {[i['markers'] for i in inf3]} · 내용 동일 {twice == once}  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    # [4] 결과가 파싱되는 파이썬인가
    try:
        ast.parse(once); parsed = True
    except SyntaxError as exc:                                   # noqa: BLE001
        parsed = False; print(f"    구문오류: {exc}")
    print(f"[4] 패치 결과 구문 검사  ", end="")
    print("OK" if parsed else "!! 실패"); bad += (not parsed)

    # [5] 고의로 망가뜨린 입력을 잡나 — anchor 를 2개로 늘린다
    broken = STUB + "\n    report={'seed':seed,'x':1}\n"
    try:
        apply(broken); caught = False
    except SystemExit:
        caught = True
    print(f"[5] anchor 2개인 입력 거부  ", end="")
    print("OK" if caught else "!! 실패 (조용히 통과했다)"); bad += (not caught)

    # [6] qpos 키가 실제로 들어갔나
    ok = "'qpos':env.data.qpos.tolist()" in once and "'joint_names'" in once
    print(f"[6] qpos·joint_names 삽입 확인  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


# ── 본체 ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--file", help="simulation/evaluate.py 경로")
    ap.add_argument("--check", action="store_true", help="상태만 보고 고치지 않는다")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.file:
        raise SystemExit("!! --file 이 필요하다")

    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 패치하지 않는다")
    print()

    p = Path(a.file).expanduser()
    if not p.is_file():
        raise SystemExit(f"!! 없다: {p}")
    text = p.read_text(encoding="utf-8")

    print(f"대상 {p}  ({len(text)} bytes)")
    for i in inspect(text):
        print(f"  {i['name']:24s} anchor {i['anchors']} / marker {i['markers']}")
    if a.check:
        print("\n--check 이므로 고치지 않는다")
        return

    # 사본에 2회 돌려 멱등성을 먼저 검산한다 (원본은 아직 건드리지 않는다)
    with tempfile.TemporaryDirectory() as d:
        t = Path(d) / p.name
        t.write_text(text, encoding="utf-8")
        once, _ = apply(t.read_text(encoding="utf-8"))
        twice, _ = apply(once)
        if once != twice:
            raise SystemExit("!! 사본 2회 적용 결과가 다르다. 멱등이 아니다. 중단")
        ast.parse(once)
    print("  사본 2회 검산 통과 (멱등 · 구문 OK)")

    bak = p.with_suffix(p.suffix + ".bak_qpos")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  백업 {bak}")

    new, notes = apply(text)
    for n in notes:
        print(f"  {n}")
    if new == text:
        print("\n변경 없음 — 이미 적용돼 있다")
        return
    ast.parse(new)
    p.write_text(new, encoding="utf-8")
    print(f"\n완료. 이제 trajectory_*.json 에 qpos 가, evaluation.json 에 joint_names 가 들어간다")


if __name__ == "__main__":
    main()
