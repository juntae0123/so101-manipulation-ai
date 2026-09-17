"""Expose a training seed on umi_adapter.train, in place.
umi_adapter.train 에 학습 시드 인자를 뚫는다. 파일을 직접 고친다.

배경: handoff 의 train.py 는 Hydra override 를 내부에서 조립하는데 시드가
없다. 그대로 3회 돌리면 "3시드 반복" 이 아니라 "비결정성 3회 반복" 이다.
둘 다 분산은 재지만, 재현이 안 된다.

설계 1: `training.seed=N` 을 **`+` 없이** 넣는다. 원본 config 에 그 키가
없으면 Hydra 가 즉시 죽는다 — 조용히 무시되는 것보다 낫다.

설계 2 (2026-09-16 정정): 멱등성 검사는 `marker` 로 한다.
첫 판은 "바뀐 문자열이 이미 있나" 로 검사했는데, 치환이 **덧붙이는** 형태면
old 가 new 안에 그대로 남아 있어서 두 번째 실행이 또 적용된다.
실제로 `--seed` 가 두 번 등록되는 코드를 만들었다.
**"적용됨" 은 적용 후에만 존재하는 표식으로 판정한다.**

Usage:
  python patch_train_seed.py --root ~/handoff [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

# (적용 후에만 존재하는 표식, 찾을 것, 바꿀 것)
EDITS: list[tuple[str, str, str]] = [
    (
        "task_path=None,seed=None):",
        "init_checkpoint=None,workers=0,lr=0.0003,task_path=None):",
        "init_checkpoint=None,workers=0,lr=0.0003,task_path=None,seed=None):",
    ),
    (
        "training.seed=",
        "    if steps is not None: overrides.append(f'training.max_train_steps={steps}')",
        "    if steps is not None: overrides.append(f'training.max_train_steps={steps}')\n"
        "    # 시드는 '+' 없이 넣는다. 원본 config 에 키가 없으면 즉시 죽어야 한다.\n"
        "    if seed is not None: overrides.append(f'training.seed={int(seed)}')",
    ),
    (
        "p.add_argument('--seed'",
        "p.add_argument('--lr',type=float,default=0.0003)",
        "p.add_argument('--lr',type=float,default=0.0003); p.add_argument('--seed',type=int)",
    ),
    (
        "a.lr,a.task,seed=a.seed)",
        "a.resume,a.init_checkpoint,a.workers,a.lr,a.task)",
        "a.resume,a.init_checkpoint,a.workers,a.lr,a.task,seed=a.seed)",
    ),
    (
        "'seed':seed,",
        "'warm_start':str(init_checkpoint) if init_checkpoint else None,",
        "'warm_start':str(init_checkpoint) if init_checkpoint else None,'seed':seed,",
    ),
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
    applied = skipped = 0
    for marker, old, new in EDITS:
        if marker in text:
            print(f"   = 이미 적용됨 ({marker})")
            skipped += 1
            continue
        n = text.count(old)
        if n == 0:
            raise SystemExit(
                f"!! 못 찾았다: {old!r}\n   원본이 다르다. 중단한다 (아무것도 안 바꿨다)")
        if n > 1:
            raise SystemExit(f"!! {n}회 나온다. 유일해야 한다: {old[:50]}...")
        text = text.replace(old, new, 1)
        print(f"   + 치환: {marker}")
        applied += 1

    ast.parse(text)          # 쓰기 전에 문법을 본다
    if applied and not args.dry_run:
        path.write_text(text, encoding="utf-8")
    # 중복 등록 자체를 검산한다 — 이 패치가 한 번 냈던 실수다
    for token, limit in [("p.add_argument('--seed'", 1), ("training.seed=", 1), ("'seed':seed,", 1)]:
        c = text.count(token)
        if c > limit:
            raise SystemExit(f"!! 중복 {c}회: {token!r}. 파일을 되돌려라")
    print(f"\n{applied}곳 적용 · {skipped}곳 이미 있음 · 문법 OK · 중복 없음"
          f"{'  (dry-run)' if args.dry_run else ''}")
    print("확인:  python -m umi_adapter.train --help | grep -- --seed")


if __name__ == "__main__":
    main()
