"""Let umi_adapter.train accept extra Hydra overrides from the CLI.
umi_adapter.train 에 임의 Hydra override 를 밖에서 넣을 수 있게 뚫는다.

배경: `task.action_horizon` 을 8 로 바꿔야 하는데 train.py 가 override 목록을
내부에서만 조립한다. 실데이터(8스텝/0.8초)와 시뮬(16스텝/1.6초)의 예측 지평을
맞추려면 이 통로가 필요하다. 표본 주기는 이미 양쪽 10Hz 다
(obs·action 둘 다 down_sample_steps=3, 30fps zarr 기준) 🟢.

설계: `--override KEY=VALUE` 를 반복 지정한다. **조립이 끝난 맨 뒤에 붙인다** —
Hydra 는 뒤에 온 것이 이긴다. 그래야 기본값을 확실히 덮는다.
`+` 없이 넣으므로 원본 config 에 없는 키는 즉시 에러가 난다. 오타가 조용히
무시되는 것보다 낫다.

멱등성: 적용 후에만 존재하는 표식(marker)으로 판정한다.
(2026-09-16 에 marker 없이 만들었다가 `--seed` 가 두 번 등록되는 코드를 냈다)

Usage:
  python patch_train_override.py --root ~/handoff [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

# (적용 후에만 존재하는 표식, 찾을 것, 바꿀 것)
EDITS: list[tuple[str, str, str]] = [
    (
        "seed=None,extra_overrides=None):",
        "task_path=None,seed=None):",
        "task_path=None,seed=None,extra_overrides=None):",
    ),
    (
        "extra_overrides or ()",
        "    env=os.environ.copy()",
        "    # 사용자가 넘긴 override 는 **맨 뒤**에 붙인다. Hydra 는 뒤가 이긴다.\n"
        "    for _ov in (extra_overrides or ()):\n"
        "        if '=' not in _ov: raise ValueError(f'override 형식이 KEY=VALUE 가 아니다: {_ov!r}')\n"
        "        overrides.append(_ov)\n"
        "    env=os.environ.copy()",
    ),
    (
        "p.add_argument('--override'",
        "p.add_argument('--seed',type=int)",
        "p.add_argument('--seed',type=int); p.add_argument('--override',action='append',default=[],"
        "help='Hydra override 를 그대로 전달한다. 반복 지정 가능 (예: --override task.action_horizon=8)')",
    ),
    (
        "extra_overrides=a.override)",
        "seed=a.seed)",
        "seed=a.seed,extra_overrides=a.override)",
    ),
]

# launch.json 기록은 특수 처리한다 — 서버에 적용된 시드 패치가 4곳짜리라
# `'seed':seed,` 가 없을 수도, 5곳짜리라 있을 수도 있다. 둘 다 받는다.
LAUNCH_ANCHOR = "'warm_start':str(init_checkpoint) if init_checkpoint else None,"

UNIQUE = [
    "p.add_argument('--override'",
    "extra_overrides=a.override)",
    "'extra_overrides':",
    "for _ov in (extra_overrides or ()):",
    "'extra_overrides':",
    "'seed':seed,",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    path = Path(args.root).expanduser().resolve() / "umi_adapter" / "train.py"
    if not path.exists():
        raise SystemExit(f"!! 없다: {path}")
    text = path.read_text(encoding="utf-8")

    if "p.add_argument('--seed'" not in text:
        raise SystemExit("!! --seed 패치가 먼저다. 이 패치는 그 위에 붙는다")

    applied = skipped = 0
    for marker, old, new in EDITS:
        if marker in text:
            print(f"   = 이미 적용됨 ({marker})")
            skipped += 1
            continue
        n = text.count(old)
        if n == 0:
            raise SystemExit(f"!! 못 찾았다: {old!r}\n   원본이 다르다. 중단 (아무것도 안 바꿨다)")
        if n > 1:
            raise SystemExit(f"!! {n}회 나온다. 유일해야 한다: {old[:60]}...")
        text = text.replace(old, new, 1)
        print(f"   + 치환: {marker}")
        applied += 1

    # launch.json 에 seed/override 를 남긴다 (실행 조건을 파일로 남기는 게 목적)
    if "'extra_overrides':" in text:
        print("   = 이미 적용됨 ('extra_overrides' launch.json)")
        skipped += 1
    else:
        if text.count(LAUNCH_ANCHOR) != 1:
            raise SystemExit(f"!! launch.json 삽입 지점이 {text.count(LAUNCH_ANCHOR)}회다. 1이어야 한다")
        add = "'seed':seed," if "'seed':seed," not in text else ""
        text = text.replace(LAUNCH_ANCHOR,
                            LAUNCH_ANCHOR + add + "'extra_overrides':list(extra_overrides or []),", 1)
        print("   + 치환: launch.json 기록" + (" (+seed)" if add else ""))
        applied += 1

    ast.parse(text)                      # 쓰기 전에 문법을 본다
    for tok in UNIQUE:                   # 중복 등록 검산
        c = text.count(tok)
        if c > 1:
            raise SystemExit(f"!! 중복 {c}회: {tok!r}. 쓰지 않는다")
    if applied and not args.dry_run:
        path.write_text(text, encoding="utf-8")
    print(f"\n{applied}곳 적용 · {skipped}곳 기존 · 문법 OK · 중복 없음"
          f"{'  (dry-run)' if args.dry_run else ''}")
    print("확인:  python -m umi_adapter.train --help | grep -- --override")


if __name__ == "__main__":
    main()
