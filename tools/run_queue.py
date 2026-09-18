"""Run pre-registered experiment configs from a queue, one item at a time.
사전등록된 실험 조건을 큐에서 꺼내 차례로 돌린다.

    python tools/run_queue.py --dry-run       # 검증만
    python tools/run_queue.py                 # 순차 실행
    python tools/run_queue.py --parallel 2    # 2개 동시 (먼저 재고 올린다)

설계 원칙 — **실행은 자동화하고 판단은 자동화하지 않는다.**

결과를 보고 다음 조건을 기계가 정하게 만들면 안 된다. n=100 의 95% 구간 반폭이
약 ±9%p 인데 기계가 조합을 훑고 최고를 고르면 우연히 좋은 것이 반드시 나온다.
그것은 측정이 아니라 다중비교 과적합이고, 이 저장소가 "게이트 기준은 결과를 보기
전에 확정한다"로 이미 막아둔 것이다. 자동화가 그 벽을 우회하는 도구가 되면 안 된다.

그래서 이 러너는 원칙을 **강제한다**:

- `prereg` 가 가리키는 사전등록 문서가 없거나 비어 있으면 **실행을 거부한다**
- 평가에 영향을 주는 경로가 커밋되지 않았으면 **실행을 거부한다** (code_sha 정직성)
- 조건은 사람이 `queue/pending/*.yaml` 에 쓴다. 러너는 큐를 비우는 일만 한다

상태 기계는 디렉터리 이동이다. 항목 파일이 어디 있는지가 곧 상태다:
`queue/pending/` → 실행 → `queue/done/` 또는 `queue/failed/`.
`queue/LEDGER.md` 에 한 줄씩 append 한다.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# 이 줄이 없으면 `PYTHONPATH` 를 세우고 부르지 않은 호출에서 policy 임포트가 깨진다.
# 2026-09-10 에 그렇게 큐가 안 돌았다 — 도구가 호출자 환경에 의존하면 안 된다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracking.exp_log import CODE_PATHS  # noqa: E402  목록 정본

import yaml  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[1]
QUEUE = AI_ROOT / "queue"
PENDING, DONE, FAILED = QUEUE / "pending", QUEUE / "done", QUEUE / "failed"
LEDGER = QUEUE / "LEDGER.md"

# 이 경로가 더러우면 결과의 code_sha 가 실제 실행 코드를 가리키지 않는다.
# 목록의 정본은 `tracking/exp_log.CODE_PATHS` 다 — 같은 목록을 두 군데 두면 갈라진다.
DIRTY_GUARD = CODE_PATHS

REQUIRED = ("name", "prereg", "kind", "data")
KINDS = ("repeat_runs",)


def _git_dirty(paths: tuple[str, ...]) -> str:
    """Tracked changes under `paths`, empty string when clean.
    `paths` 아래의 추적 중인 변경. 깨끗하면 빈 문자열."""
    out = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths],
        cwd=AI_ROOT, capture_output=True, text=True, check=False,
    )
    return out.stdout.strip()


def _git_rev() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=AI_ROOT, capture_output=True, text=True, check=False,
    )
    return out.stdout.strip() or "unknown"


def validate(item: dict, path: Path) -> list[str]:
    """Reasons this item must not run. Empty list means it may.
    이 항목을 돌리면 안 되는 이유들. 빈 목록이면 실행 가능."""
    bad: list[str] = []
    for key in REQUIRED:
        if not item.get(key):
            bad.append(f"필수 키 없음: {key}")
    if bad:
        return bad

    if item["kind"] not in KINDS:
        bad.append(f"kind 를 모른다: {item['kind']!r} (지원: {KINDS})")

    prereg = AI_ROOT / str(item["prereg"])
    if not prereg.exists():
        bad.append(f"사전등록 문서가 없다: {item['prereg']}")
    elif prereg.stat().st_size < 200:
        bad.append(f"사전등록 문서가 200바이트 미만이다 (내용 없음): {item['prereg']}")

    data = AI_ROOT / str(item["data"])
    if not data.exists():
        bad.append(f"데이터가 없다: {item['data']}")

    if item["name"] != path.stem:
        bad.append(f"name({item['name']!r}) 과 파일명({path.stem!r}) 이 다르다")

    # 행동공간 오타를 밤새 돌린 뒤에 알면 하룻밤이 날아간다. 여기서 막는다.
    space = item.get("action_space")
    if space:
        try:
            from policy.bc import ACTION_SPACES
        except Exception as exc:  # noqa: BLE001 — 임포트 실패도 검증 실패로 본다
            bad.append(f"policy.bc 를 임포트할 수 없어 action_space 를 검증 못 한다: {exc}")
        else:
            if space not in ACTION_SPACES:
                bad.append(
                    f"action_space 를 모른다: {space!r} (지원: {ACTION_SPACES})"
                )

    # 카메라 이름 오타도 밤새 돌린 뒤에 알면 하룻밤이 날아간다.
    # 여기서는 형식만 본다 — 실제 존재 여부는 EpisodeDataset 이 첫 에피소드에서 막는다.
    cams = item.get("cameras")
    if cams is not None:
        names = [c.strip() for c in str(cams).split(",") if c.strip()]
        if not names:
            bad.append(f"cameras 가 비었다: {cams!r}")
        unknown = [c for c in names if not c.startswith("cam_")]
        if unknown:
            bad.append(f"카메라 이름이 'cam_' 으로 시작하지 않는다: {unknown}")

    # 청크 길이 오타를 밤새 돌린 뒤에 알면 하룻밤이 날아간다.
    ch = item.get("chunk")
    if ch is not None:
        try:
            ch = int(ch)
        except (TypeError, ValueError):
            bad.append(f"chunk 가 정수가 아니다: {item['chunk']!r}")
        else:
            if ch < 1:
                bad.append(f"chunk 는 1 이상이어야 한다: {ch}")

    return bad


def build_cmd(item: dict) -> list[str]:
    """The command this item runs. Only pre-registered knobs are exposed.
    이 항목이 실행할 명령. 사전등록된 손잡이만 노출한다."""
    cmd = [
        sys.executable, "tools/repeat_runs.py",
        "--data", str(item["data"]),
        "--runs", str(item.get("runs", 3)),
        "--episodes", str(item.get("episodes", 100)),
        "--seed-base", str(item.get("seed_base", 0)),
        "--eval-seed-base", str(item.get("eval_seed_base", 3000)),
        "--tag", str(item["name"]),
        "--device", str(item.get("device", "cuda")),
        "--policy-device", str(item.get("policy_device", "cpu")),
        "--log",
    ]
    if item.get("epochs") is not None:
        cmd += ["--epochs", str(item["epochs"])]
    if item.get("action_space"):
        cmd += ["--action-space", str(item["action_space"])]
    if item.get("cameras"):
        cmd += ["--cameras", str(item["cameras"])]
    if item.get("image_noise") is not None:
        cmd += ["--image-noise", str(item["image_noise"])]
    # 학습 타깃 사이드카. 계약 npz 는 그대로 두고 타깃만 바꾼다 (S15P21A103-170).
    if item.get("target_sidecar"):
        cmd += ["--target-sidecar", str(item["target_sidecar"])]
    # 행동 청크 길이. 1 이면 기존 BC 와 비트 동일 (S15P21A103-171).
    if item.get("chunk") is not None and int(item["chunk"]) != 1:
        cmd += ["--chunk", str(int(item["chunk"]))]
    return cmd


def note(line: str) -> None:
    """Append one line to the ledger. Never rewrites history.
    원장에 한 줄 append 한다. 과거를 다시 쓰지 않는다."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text(
            "# 큐 원장 (append only)\n\n"
            "| 시각 | 항목 | git | rc | 뜻 | 로그 |\n|---|---|---|---:|---|---|\n",
            encoding="utf-8",
        )
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_item(path: Path, log_dir: Path, slot: int | None = None,
             gpu: int | None = None) -> tuple[Path, int, str]:
    """Run one queue item to completion. Returns (path, returncode, log path).
    큐 항목 하나를 끝까지 돌린다. (경로, 종료코드, 로그경로) 를 반환한다.

    `slot` 은 이 항목이 쓸 GPU 를 정한다. 없으면 `runtime_limits.pick_gpu()` 가 고른다.

    ⚠️ 동시에 여러 항목을 띄우면 `pick_gpu()` 로는 안 된다 — 부하를 **시작 시점에
    한 번** 읽으므로, 같은 순간에 뜬 항목들은 아직 아무 장도 바쁘지 않아 전부 같은
    장을 고른다. 그래서 큐가 슬롯을 명시로 나눈다."""
    item = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = item["name"]
    log = log_dir / f"{name}.log"
    env = dict(os.environ)
    # 항목마다 다른 락 이름을 줘야 병렬 기동이 가능하다 (runtime_limits.claim).
    env["AI_CLAIM_NAME"] = f"queue_{name}"
    # `--gpu` 로 한 장에 몰 수 있다. 잡당 808MiB 라 32GB 카드에 여러 개가 들어간다
    # (2026-09-15 실측 🟢). 한 장에 모으면 나머지 할당분을 건드리지 않는다.
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    elif slot is not None:
        from runtime_limits import ALLOWED_GPUS
        env["CUDA_VISIBLE_DEVICES"] = str(ALLOWED_GPUS[slot % len(ALLOWED_GPUS)])
    env.setdefault("AI_THREADS", "2")
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONPATH"] = f"{AI_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    cmd = build_cmd(item)
    with log.open("w", encoding="utf-8") as fh:
        fh.write(f"# {name}\n# git {_git_rev()}\n# {' '.join(cmd)}\n\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=AI_ROOT, env=env,
                            stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
    return path, rc, str(log.relative_to(AI_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=1,
                        help="동시 실행 수. 학습은 GPU 지만 평가는 CPU+EGL 렌더라 "
                             "경합은 CPU 에서 난다. 2 로 먼저 재고 올린다")
    parser.add_argument(
        "--gpu", type=int, default=None, metavar="N",
        help="모든 항목을 이 카드 하나에 올린다. 없으면 할당분에 라운드로빈. "
             "잡당 808MiB 라 32GB 카드에 8잡까지 여유가 있다 (2026-09-15 실측)",
    )
    parser.add_argument("--dry-run", action="store_true", help="검증만 하고 끝낸다")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="더러운 트리 거부를 푼다. 결과의 code_sha 가 "
                             "실제 실행 코드를 가리키지 않게 된다 — 리허설 전용")
    args = parser.parse_args()

    for d in (PENDING, DONE, FAILED):
        d.mkdir(parents=True, exist_ok=True)

    items = sorted(PENDING.glob("*.yaml"))
    if not items:
        print(f"큐가 비어 있다: {PENDING.relative_to(AI_ROOT)}/*.yaml")
        return 0

    print(f"큐 {len(items)}건 · git {_git_rev()} · 병렬 {args.parallel}\n")

    # 검증을 **전부 먼저** 한다. 하나라도 막히면 아무것도 시작하지 않는다.
    # 반쯤 돌다 멈추면 어느 조건이 어느 트리에서 돌았는지 뒤섞인다.
    blocked = False
    plans: list[Path] = []
    for path in items:
        try:
            item = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            print(f"✗ {path.name}: YAML 오류 — {exc}")
            blocked = True
            continue
        if not isinstance(item, dict):
            print(f"✗ {path.name}: 최상위가 매핑이 아니다")
            blocked = True
            continue
        reasons = validate(item, path)
        if reasons:
            print(f"✗ {path.name}")
            for r in reasons:
                print(f"    {r}")
            blocked = True
            continue
        print(f"✓ {path.name}  →  {' '.join(build_cmd(item)[1:])}")
        plans.append(path)

    dirty = _git_dirty(DIRTY_GUARD)
    if dirty and not args.allow_dirty:
        print("\n✗ 커밋되지 않은 변경이 있다. 결과의 code_sha 가 실제 실행 코드를 "
              "가리키지 않는다. 커밋하고 다시 실행하라:")
        for line in dirty.splitlines():
            print(f"    {line}")
        return 2

    if blocked:
        print("\n✗ 막힌 항목이 있어 아무것도 실행하지 않았다.")
        return 2
    if args.dry_run:
        print(f"\n검증만 했다. {len(plans)}건 실행 가능.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = AI_ROOT / "out" / f"queue_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    rev = _git_rev()

    def finish(path: Path, rc: int, log: str) -> None:
        # rc=1 은 repeat_runs 의 **배포 게이트 실패**다. 실행 오류가 아니다.
        meaning = {0: "게이트 통과", 1: "게이트 실패(정상 종료)"}.get(rc, "실행 오류")
        dest = (DONE if rc in (0, 1) else FAILED) / path.name
        shutil.move(str(path), str(dest))
        when = datetime.now().strftime("%m-%d %H:%M")
        note(f"| {when} | {path.stem} | {rev} | {rc} | {meaning} | `{log}` |")
        print(f"[{when}] {path.stem}: rc={rc} {meaning} → {dest.parent.name}/  ({log})")

    started = time.time()
    from runtime_limits import ALLOWED_GPUS
    if args.gpu is not None and args.gpu not in ALLOWED_GPUS:
        print(f"✗ --gpu {args.gpu} 는 할당분 {list(ALLOWED_GPUS)} 밖이다. "
              f"남의 작업을 밀어낸다.")
        return 2
    if args.gpu is not None:
        mib = args.parallel * 808
        print(f"· 전 항목을 GPU {args.gpu} 에 올린다 "
              f"(동시 {args.parallel}잡 ≈ {mib}MiB / 32GB)")
    if args.parallel <= 1:
        for path in plans:
            finish(*run_item(path, log_dir, gpu=args.gpu))
    else:
        if args.gpu is None and args.parallel > len(ALLOWED_GPUS):
            print(f"⚠️ --parallel {args.parallel} 이 할당분 {len(ALLOWED_GPUS)}장보다 많다. "
                  f"같은 장에 두 항목이 올라간다")
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            jobs = list(enumerate(plans))
            for path, rc, log in pool.map(
                lambda t: run_item(t[1], log_dir, t[0], args.gpu), jobs
            ):
                finish(path, rc, log)

    mins = (time.time() - started) / 60
    print(f"\n큐 소진 {len(plans)}건 · {mins:.1f}분 · 원장 {LEDGER.relative_to(AI_ROOT)}")
    print("판정은 사람이 한다. 각 항목의 사전등록 문서를 열고 대조하라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
