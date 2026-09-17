"""Split commanded lift target from success lift threshold, in place.
명령 리프트 목표와 성공 판정 리프트 임계를 분리한다. 파일을 직접 고친다.

배경: configs 의 lift_height(0.13) 는 IK 목표이고, 성공 판정은 expert.py 와
evaluate.py 에 0.1 로 **하드코딩**돼 있었다. 같은 이름처럼 보여서 0.13 을
성공 기준으로 읽으면 2/100 이 나오고 0.10 으로 읽으면 100/100 이 나온다.
트랙 A 합의: commanded = 0.13 유지, success_min_lift_m = 0.10 을 명시한다.

설계 원칙: 레거시 체크포인트에는 success_min_lift_m 이 없다. 조용히 0.1 로
떨어지면 "설정된 0.10" 과 "설정이 없어서 0.10" 이 같은 모양이 된다.
그래서 출처를 리포트에 남긴다 (success_min_lift_source).

Usage:
  python patch_lift_threshold.py --root ~/handoff [--dry-run]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

SUCCESS_DEFAULT = 0.10

# (파일, 찾을 문자열, 바꿀 문자열, 최소 등장 횟수)
EDITS: list[tuple[str, str, str, int]] = [
    (
        "simulation/evaluate.py",
        "z>initial_z+0.1 and coupled",
        "z>initial_z+success_min_lift and coupled",
        1,
    ),
    (
        "simulation/expert.py",
        "z>env.initial_z+0.1 and sum(env.data.qpos[env.fids])<0.06",
        "z>env.initial_z+_success_min_lift(env) and sum(env.data.qpos[env.fids])<0.06",
        1,
    ),
    (
        "simulation/expert.py",
        "z>env.initial_z+.1 and coupling<.025",
        "z>env.initial_z+_success_min_lift(env) and coupling<.025",
        1,
    ),
]

EXPERT_HELPER = '''

def _success_min_lift(env) -> float:
    """Success lift threshold in metres, from task config.
    성공 판정 리프트 임계값[m]. task 설정에서 읽는다.

    lift_height 는 IK **명령** 목표라 성공 기준이 아니다. 둘을 같은 값으로
    읽으면 전문가 자신이 도달 못 하는 기준으로 채점하게 된다.
    """
    v = env.task.get("success_min_lift_m")
    if v is None:
        return 0.10  # legacy. 호출부가 출처를 기록한다
    return float(v)
'''

EVALUATE_RESOLVER = '''    # 성공 판정 리프트 임계값. 없으면 레거시 0.10 이지만 **출처를 기록한다.**
    # 기록이 없으면 "설정된 0.10" 과 "설정이 없는 0.10" 이 구분되지 않는다.
    _sml = env.task.get('success_min_lift_m')
    success_min_lift = 0.10 if _sml is None else float(_sml)
    success_min_lift_source = 'legacy_default' if _sml is None else 'task_config'
    if _sml is None:
        print('!! success_min_lift_m 이 task 에 없다. 레거시 0.10 으로 채점한다. '
              '설정본으로 채점하려면 --task 로 config 를 넘겨라', flush=True)
'''

CONFIG_LINE = (
    "# lift_height 는 IK **명령** 목표다. 성공 판정 기준이 아니다.\n"
    "# 성공 판정은 아래 success_min_lift_m 을 쓴다 (트랙 A·B 합의 2026-09-16).\n"
    "success_min_lift_m: 0.10\n"
)


def patch_text(path: Path, old: str, new: str, need: int, dry: bool) -> int:
    """Replace old with new, refusing to pass silently when nothing matched.
    old 를 new 로 바꾼다. 하나도 못 찾으면 조용히 넘어가지 않고 실패시킨다."""
    text = path.read_text(encoding="utf-8")
    hits = text.count(old)
    if hits == 0:
        if text.count(new) > 0:
            print(f"   = 이미 적용됨: {path.name} :: {old[:40]}...")
            return 0
        raise SystemExit(
            f"!! 못 찾았다: {path}\n   찾던 문자열: {old!r}\n"
            f"   원본이 다르다. 손으로 확인해라 (패치 중단, 아무것도 안 바꿨다)"
        )
    if hits < need:
        raise SystemExit(f"!! {path}: {hits}회만 찾았다 (최소 {need})")
    if not dry:
        path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"   + {path.name}: {hits}곳 치환")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="handoff 패키지 루트")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not (root / "simulation" / "evaluate.py").exists():
        raise SystemExit(f"!! handoff 루트가 아니다: {root}")

    dry = args.dry_run
    print(f"root = {root}{'  (dry-run)' if dry else ''}")

    # 1) configs
    print("[1] configs")
    cfgs = sorted((root / "configs").glob("*.yaml"))
    touched = 0
    for c in cfgs:
        t = c.read_text(encoding="utf-8")
        if "lift_height" not in t:
            continue
        if "success_min_lift_m" in t:
            print(f"   = 이미 있음: {c.name}")
            continue
        if not dry:
            c.write_text(t.rstrip("\n") + "\n" + CONFIG_LINE, encoding="utf-8")
        print(f"   + {c.name}: success_min_lift_m {SUCCESS_DEFAULT} 추가")
        touched += 1
    if touched == 0 and not any("success_min_lift_m" in c.read_text(encoding="utf-8") for c in cfgs):
        raise SystemExit("!! lift_height 를 가진 config 가 하나도 없다. 경로가 틀렸다")

    # 2) expert.py 헬퍼
    print("[2] simulation/expert.py 헬퍼")
    ep = root / "simulation" / "expert.py"
    et = ep.read_text(encoding="utf-8")
    if "_success_min_lift" not in et:
        marker = "import numpy as np"
        if marker not in et:
            raise SystemExit("!! expert.py 에서 import 지점을 못 찾았다")
        if not dry:
            ep.write_text(et.replace(marker, marker + EXPERT_HELPER, 1), encoding="utf-8")
        print("   + _success_min_lift() 삽입")
    else:
        print("   = 이미 있음")

    # 3) evaluate.py 해석부
    print("[3] simulation/evaluate.py 해석부")
    vp = root / "simulation" / "evaluate.py"
    vt = vp.read_text(encoding="utf-8")
    if "success_min_lift_source" not in vt:
        anchor = "    tf=get_image_transform("
        if anchor not in vt:
            raise SystemExit("!! evaluate.py 에서 삽입 지점(tf=get_image_transform)을 못 찾았다")
        if not dry:
            vp.write_text(vt.replace(anchor, EVALUATE_RESOLVER + anchor, 1), encoding="utf-8")
        print("   + success_min_lift 해석부 삽입")
    else:
        print("   = 이미 있음")

    # 4) 판정식 치환
    print("[4] 판정식")
    for rel, old, new, need in EDITS:
        patch_text(root / rel, old, new, need, dry)

    # 5) 리포트 기록
    print("[5] 리포트 필드")
    vt = vp.read_text(encoding="utf-8")
    if "report['success_min_lift_m']" not in vt:
        anchor = "            report['action_steps']=action_steps"
        if anchor not in vt:
            raise SystemExit("!! evaluate.py 에서 리포트 삽입 지점을 못 찾았다")
        add = (anchor
               + "\n            report['success_min_lift_m']=success_min_lift"
               + "\n            report['success_min_lift_source']=success_min_lift_source"
               + "\n            report['commanded_lift_height_m']=env.task.get('lift_height')")
        if not dry:
            vp.write_text(vt.replace(anchor, add, 1), encoding="utf-8")
        print("   + success_min_lift_m / _source / commanded_lift_height_m 기록")
    else:
        print("   = 이미 있음")

    print("\n끝. 확인:")
    print(f"  python -c \"import ast,sys;[ast.parse(open(p).read()) for p in "
          f"['{root}/simulation/evaluate.py','{root}/simulation/expert.py']];print('문법 OK')\"")


if __name__ == "__main__":
    main()
