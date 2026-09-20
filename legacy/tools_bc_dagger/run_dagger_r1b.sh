#!/usr/bin/env bash
# DAgger Round 1b — 대행 AI 가 작성한 실행 블록을 그대로 파일로 옮긴 것.
# 내용은 변형하지 않았다. 터미널 붙여넣기에서 heredoc 이 잘려서 파일로 옮겼을 뿐이다.
#
#   # [서버]
#   bash tools/run_dagger_r1b.sh

set -e
set -o pipefail
cd "$(dirname "$0")/.."

export AI_THREADS=2
export MUJOCO_GL=egl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m py_compile tools/collect_dagger_segments.py
python tools/collect_dagger_segments.py --help >/dev/null

STAMP=$(date +%Y%m%d_%H%M%S)
LABELS="out/dagger_segments_${STAMP}"
MERGED="out/dagger_merged_${STAMP}"
TRAIN_OUT="out/dagger_train_${STAMP}"
COLLECT_LOG="out/dagger_segments_${STAMP}.log"

mkdir -p out

set +e
python tools/collect_dagger_segments.py \
  --policy-ckpt checkpoints/bc/sim_pick_v5_seed0.pt \
  --policy-ckpt checkpoints/bc/sim_pick_v5_seed1.pt \
  --policy-ckpt checkpoints/bc/sim_pick_v5_seed2.pt \
  --episodes 100 \
  --seed-base 4000 \
  --label-start-tick 30 \
  --out "$LABELS" \
  --log \
  2>&1 | tee "$COLLECT_LOG"
COLLECT_RC=${PIPESTATUS[0]}
set -e

test -f "$LABELS/dagger_segments_result.json"

python - "$LABELS/dagger_segments_result.json" "$COLLECT_LOG" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
doc = [
    "# MEASURE — DAgger Round 1b valid segments (2026-09-09)",
    "",
    "- 확신도: 🟢 실행·로그 확인",
    f"- 원본 로그: `{sys.argv[2]}`",
    "",
    "## 수집 결과",
    "",
    f"- queries: {result['queries']}",
    f"- valid labels: {result['valid_ticks']} "
    f"({100*result['valid_label_rate']:.2f}%)",
    f"- stored labels: {result['stored_ticks']} "
    f"({100*result['stored_label_rate']:.2f}%)",
    f"- segments: {result['segments']}",
    f"- invalid IK ticks: {result['invalid_ik_ticks']}",
    f"- invalid range ticks: {result['invalid_range_ticks']}",
    f"- invalid both ticks: {result['invalid_both_ticks']}",
    f"- max state excess: {result['max_state_excess']:.8f}",
    f"- max action excess: {result['max_action_excess']:.8f}",
    f"- contract violations: {result['contract_violations']}",
    f"- gates: {result['gates']}",
]
Path("docs/MEASURE_dagger_segments_0909.md").write_text(
    "\n".join(doc), encoding="utf-8"
)
PY

TRAIN_GO=$(
  python - "$LABELS/dagger_segments_result.json" <<'PY'
import json
import sys
print("1" if json.load(open(sys.argv[1]))["gates"]["train_go"] else "0")
PY
)

if test "$TRAIN_GO" != "1"; then
  printf '\n===== 수집 게이트 실패 =====\n'
  cat docs/MEASURE_dagger_segments_0909.md
  exit "$COLLECT_RC"
fi

python - "datasets/sim_pick_v5" "$LABELS" "$MERGED" <<'PY'
import os
import shutil
import sys
from pathlib import Path

from contract.episode import read_episode, validate, write_dataset_index

demo, dagger, out = map(Path, sys.argv[1:])
if out.exists():
    raise FileExistsError(out)
out.mkdir(parents=True)

for source in (demo, dagger):
    for npz in sorted(source.glob("*.npz")):
        for src in (npz, npz.with_suffix(".json")):
            dst = out / src.name
            if dst.exists():
                raise FileExistsError(dst)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

violations = sum(
    len(validate(read_episode(path)))
    for path in sorted(out.glob("*.npz"))
)
if violations:
    raise RuntimeError(f"merged contract violations: {violations}")

write_dataset_index(
    out,
    extra={
        "experimental_only": True,
        "sources": [str(demo), str(dagger)],
        "contract_violations": violations,
    },
)
print(
    f"merged episodes={len(list(out.glob('*.npz')))}, "
    "contract violations=0"
)
PY

mkdir -p "$TRAIN_OUT"

PYTHONUNBUFFERED=1 python tools/repeat_runs.py \
  --data "$MERGED" \
  --runs 3 \
  --episodes 100 \
  --epochs 30 \
  --seed-base 0 \
  --eval-seed-base 3000 \
  --tag dagger_segments_0909 \
  --device cuda \
  --log \
  2>&1 | tee "$TRAIN_OUT/repeat_runs.log"

{
  printf '\n\n## 3회 학습·롤아웃\n\n```text\n'
  grep -E '시드|평균|범위|합산|배포 게이트|→ 배포|bc[[:space:]]+[0-9]+/100' \
    "$TRAIN_OUT/repeat_runs.log" || true
  printf '```\n'
} >> docs/MEASURE_dagger_segments_0909.md

python tracking/findings.py

printf '\n===== 최종 결과 =====\n'
cat docs/MEASURE_dagger_segments_0909.md
printf '\n===== 변경 파일 =====\n'
git status --short
