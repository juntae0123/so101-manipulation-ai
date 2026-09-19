"""Bring the five skill registry entries in line with what we measured.
스킬 레지스트리 5개를 실측에 맞춘다.

Why this exists / 왜 필요한가
The registry was written 2026-09-02 and still says `dof: 6`, `action_space:
joint_delta`, an empty `ckpt_uri`, a workspace of x[0.10, 0.25] and an object
size of 15~25mm. Every one of those is contradicted by a measurement we have
since taken, and the registry is what the frontend, the BE contract and the VLM
output schema all derive from. A stale contract file is the quiet kind of wrong.
레지스트리는 0902 작성본이고 아직 `dof: 6`, `action_space: joint_delta`,
빈 `ckpt_uri`, 작업영역 x[0.10, 0.25], 물체 15~25mm 를 담고 있다. 전부 이후
실측과 어긋난다. 프런트·BE 계약·VLM 출력 스키마가 전부 이 파일에서 파생되므로
낡은 계약 파일은 조용히 틀리는 쪽이다.

What it does NOT do / 하지 않는 것
It does not flip `status` to deployed. Placing (the scripted half) is 0/5 and
there is no real-robot rollout number, so every skill stays `planned`.
`status` 를 deployed 로 바꾸지 않는다. 놓기 스크립트가 0/5 이고 실물 롤아웃
수치가 없다. 다섯 개 전부 `planned` 로 남는다.

Idempotent: running it twice changes nothing the second time, and it says so.
멱등: 두 번 돌리면 두 번째에는 아무것도 바뀌지 않고, 그렇다고 말한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REG = Path(__file__).resolve().parents[1] / "registry" / "skills"

# Measured values, each with the measurement that fixed it.
# 실측값. 각각 근거 측정을 같이 적는다.
CKPT_URI = "outputs/f0918_B_42/checkpoints/latest.ckpt"
CKPT_SHA = "abb8a77d13805c2b41dd99d88df12855aa5b3ff6e747afec6975c1c1798ad766"
TRAINED_ON = "umi_real_relative_20260911_v10 74편 중 train 60편 (분할시드 42) · warm start e2_C"
ACTION_SPACE = "eef_relative_rot6d"
CONTRACT_VERSION = "umi_official/0.1.0"

PATCH_NOTES = {
    "plan": "A — 파지 정책 1개를 5개가 공유. B(스킬별 독립 정책)는 확장 목표",
    "blocked_by": (
        "놓기 스크립트 0/5 · 실물 롤아웃 0건(1a·1b 안전절차 승인 대기) · "
        "5스킬 대상물 미확정(HW)"
    ),
    "action_space_meaning": (
        "T_next = T_cur @ A_relative · 회전은 행렬의 첫 두 '행' · gap 절대값 m · "
        "10Hz 8점 · 실행 구간 index 1..4 (0-based)"
    ),
    "ckpt_sha256": CKPT_SHA,
    "_updated_0919": (
        "dof 6→5(실측 5자유도+그리퍼) · action_space joint_delta→eef_relative_rot6d · "
        "workspace 도달 포락선 실측으로 교체 · object_size_mm 기획서값 폐기(대상물 미확정) · "
        "근거 MEASURE_folds_real_0919 · DECISIONS_AI_0918_model_and_be(D-AI-60/61)"
    ),
}

# Reach envelope, side grasp, handoff SO-101 base_height 0. Conditions travel with the number.
# 도달 포락선(측면 파지, handoff SO-101, base_height 0). 조건을 숫자와 함께 남긴다.
WORKSPACE = {
    "x": [0.34, 0.46],
    "y": [-0.175, 0.175],
    "_conditions": "측면 파지 · handoff SO-101 · base_height 0 · z 0.027~0.051 구간 실측. "
                   "top-down 은 z 0.02~0.04 만 도달(513칸 중 28칸)이라 상면 파지 불가",
}


def patch(doc: dict) -> tuple[dict, list[str]]:
    """Apply the measured values. Return the new doc and what actually changed.
    실측값을 적용하고 실제로 바뀐 것만 돌려준다."""
    changed: list[str] = []

    def setv(container: dict, key: str, value: object, label: str) -> None:
        if container.get(key) != value:
            changed.append(f"{label}: {container.get(key)!r} -> {value!r}")
            container[key] = value

    pol = doc.setdefault("policy", {})
    setv(pol, "action_space", ACTION_SPACE, "policy.action_space")
    setv(pol, "ckpt_uri", CKPT_URI, "policy.ckpt_uri")
    setv(pol, "trained_on", TRAINED_ON, "policy.trained_on")
    setv(pol, "contract_version", CONTRACT_VERSION, "policy.contract_version")

    rob = doc.setdefault("robot", {})
    setv(rob, "dof", 5, "robot.dof")

    setv(doc, "workspace_m", WORKSPACE, "workspace_m")
    # 빈 리스트로 둔다. validate() 가 "두 값이어야 한다"로 명시적으로 걸어준다.
    # None 은 로더를 깨뜨리고, 옛 값을 남기면 미확정이 확정처럼 보인다.
    setv(doc, "object_size_mm", [], "object_size_mm")

    notes = doc.setdefault("notes", {})
    for k, v in PATCH_NOTES.items():
        setv(notes, k, v, f"notes.{k}")

    return doc, changed


def selftest() -> int:
    """Rows with a known answer, including one deliberately wrong input.
    정답을 아는 행. 고의로 틀린 입력 한 줄을 포함한다."""
    ok = 0
    total = 0

    def check(label: str, cond: bool) -> None:
        nonlocal ok, total
        total += 1
        if cond:
            ok += 1
            print(f"  OK  [{total}] {label}")
        else:
            print(f"  !!  [{total}] {label}")

    stale = {"policy": {"action_space": "joint_delta", "ckpt_uri": ""},
             "robot": {"dof": 6}, "object_size_mm": [15.0, 25.0]}
    out, ch = patch(json.loads(json.dumps(stale)))
    check("낡은 항목을 고치면 변경이 잡힌다", len(ch) >= 5)
    check("dof 가 5 가 된다", out["robot"]["dof"] == 5)
    check("object_size_mm 은 지어내지 않고 비운다", out["object_size_mm"] == [])
    check("status 를 건드리지 않는다", "status" not in out)

    out2, ch2 = patch(json.loads(json.dumps(out)))
    check("멱등: 두 번째 실행은 변경 0", ch2 == [])

    # Deliberately wrong input: already-correct doc must NOT report a change.
    # 고의 오답 행: 이미 맞는 문서에 변경이 잡히면 판별력이 없는 것이다.
    bad = json.loads(json.dumps(out))
    bad["robot"]["dof"] = 6
    _, ch3 = patch(bad)
    check("판별력: dof 만 되돌려 놓으면 그 한 줄만 잡힌다",
          len(ch3) == 1 and ch3[0].startswith("robot.dof"))

    print(f"자체검증 {ok} / {total}")
    return 0 if ok == total else 1


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()

    files = sorted(REG.glob("*.json"))
    print(f"레지스트리 파일 {len(files)} / 기대 5")
    if len(files) != 5:
        print("!! 파일 수가 5가 아니다. 범위가 불완전하다. 중단")
        return 2

    touched = 0
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        doc, changed = patch(doc)
        if changed:
            touched += 1
            f.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
        print(f"\n{f.name}  변경 {len(changed)}건")
        for c in changed:
            print(f"    {c}")
    print(f"\n바뀐 파일 {touched} / {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
