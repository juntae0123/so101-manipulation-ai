#!/usr/bin/env bash
# 대행 AI 작성. 원문 그대로 파일로 옮긴 것 — 내용 변형 없음.
#
#   # [서버]
#   bash tools/run_freeze_dagger_parallel.sh

set -e
set -o pipefail
cd ~/S15P21A103/AI

export AI_THREADS=2
export MUJOCO_GL=egl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="out/freeze_dagger_parallel_${STAMP}"
DOC="docs/MEASURE_freeze_dagger_parallel_0910.md"
mkdir -p "$OUT"

cat > docs/PREREG_freeze_dagger_parallel_0910.md <<'MD'
# PREREG — DAgger post-close freeze probe (2026-09-10)

- 목적: DAgger 정책이 물체 근처에서 닫은 뒤에도 실패하는 이유를 측정
- checkpoints: 최신 dagger_segments seed0/1/2
- n=100/checkpoint
- 평가 seeds: 3100~3199
- policy device: cpu
- render: true
- 기록: freeze, jaw contact, lift, close 여부
- gripper schedule probe와 다른 시드 블록이므로 직접 paired 비교하지 않는다.
- mean_max_lift는 gripper probe가 200틱 전체를 실행하므로 기존 조기종료 rollout 수치와 직접 비교하지 않는다.
MD

mapfile -t CKPTS < <(
  find checkpoints/bc -maxdepth 1 \
    -name 'dagger_merged_*_dagger_segments_0909_seed[012].pt' |
    sort |
    tail -3
)

test "${#CKPTS[@]}" -eq 3

HELP="$OUT/help.txt"
python tools/probe_freeze.py --help 2>&1 | tee "$HELP"

for CKPT in "${CKPTS[@]}"; do
  NAME=$(basename "$CKPT" .pt)
  LOG="$OUT/${NAME}.log"
  ARGS=(--policy-ckpt "$CKPT")

  grep -qF -- '--episodes' "$HELP" && ARGS+=(--episodes 100)
  grep -qF -- '--seed-base' "$HELP" && ARGS+=(--seed-base 3100)
  if grep -qF -- '--policy-device' "$HELP"; then
    ARGS+=(--policy-device cpu)
  elif grep -qF -- '--device' "$HELP"; then
    ARGS+=(--device cpu)
  fi
  grep -qF -- '--render' "$HELP" && ARGS+=(--render)
  grep -qF -- '--log' "$HELP" && ARGS+=(--log)

  printf '\n===== %s =====\n' "$NAME"

  if grep -q 'runtime_limits' tools/probe_freeze.py; then
    python tools/probe_freeze.py "${ARGS[@]}" 2>&1 | tee "$LOG"
  else
    python - tools/probe_freeze.py "${ARGS[@]}" <<'PY' 2>&1 | tee "$LOG"
import runpy
import sys
import runtime_limits

script = sys.argv[1]
args = sys.argv[2:]
runtime_limits.claim("probe_freeze_parallel")
runtime_limits.torch_threads()
sys.argv = [script, *args]
runpy.run_path(script, run_name="__main__")
PY
  fi
done

{
  printf '# MEASURE — DAgger post-close freeze probe (2026-09-10)\n\n'
  printf -- '- 확신도: 🟢 실행·로그 확인\n'
  printf -- '- 조건: n=100/checkpoint, seeds 3100~3199, render=True, policy CPU\n'
  printf -- '- 사용자 승인 병렬 작업. runtime_limits 자동 GPU 할당, 직접 GPU 지정 없음.\n\n'

  for LOG in "$OUT"/*.log; do
    printf '## %s\n\n```text\n' "$(basename "$LOG" .log)"
    grep -Ei \
      'freeze|동결|close|닫|contact|접촉|lift|상승|success|성공|phase|파지|게이트|판정' \
      "$LOG" || true
    printf '```\n\n'
  done
} > "$DOC"

printf '\n===== 병렬 실험 결과 =====\n'
cat "$DOC"
printf '\n원본 로그: AI/%s\n' "$OUT"
