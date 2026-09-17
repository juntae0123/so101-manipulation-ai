#!/usr/bin/env bash
# E1 학습 3시드 병렬 런처 (서버 V100)
# PREREG_3day_0916.md · 학습 시드 0/1/2 · GPU 2/3/4
#
# 쓰는 법:
#   EPOCHS=40 bash run_e1_train.sh
#   EPOCHS 는 1 epoch 계측 후에 정한다. 기본값을 주지 않는다 —
#   "안 정한 것" 과 "정한 것" 이 같은 모양이 되면 안 된다.
set -euo pipefail

ROOT="${ROOT:-$HOME/handoff}"
PY="${PY:-$HOME/envs/handoff312/bin/python}"
DATASET="${DATASET:-$ROOT/outputs/ds_6000.zarr.zip}"
TASK="${TASK:-$ROOT/configs/can_side.yaml}"
TAG="${TAG:-e1}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-4}"
: "${EPOCHS:?EPOCHS 를 지정해라 (예: EPOCHS=40 bash run_e1_train.sh)}"

GPUS=(2 3 4)
SEEDS=(0 1 2)
MEM_FREE_LIMIT_MIB=500

die() { echo "!! $*" >&2; exit 1; }

# ── 착수 전 검사. 하나라도 안 맞으면 아무것도 안 띄운다 ────────────
[ -x "$PY" ]            || die "python 이 없다: $PY"
[ -e "$DATASET" ]       || die "데이터셋이 없다: $DATASET"
[ -f "$TASK" ]          || die "task config 가 없다: $TASK"
[ -f "$ROOT/umi_adapter/train.py" ] || die "handoff 루트가 아니다: $ROOT"

grep -q "p.add_argument('--seed'" "$ROOT/umi_adapter/train.py" \
  || die "train.py 에 --seed 가 없다. 시드 패치를 먼저 적용해라"

command -v nvidia-smi >/dev/null || die "nvidia-smi 가 없다"
for g in "${GPUS[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null) \
    || die "GPU $g 를 조회 못 한다"
  [ -n "$used" ] || die "GPU $g 메모리 조회가 빈 값이다"
  if [ "$used" -gt "$MEM_FREE_LIMIT_MIB" ]; then
    die "GPU $g 가 이미 ${used}MiB 쓰고 있다. 남의 잡 위에 얹지 않는다"
  fi
done

[ "${#GPUS[@]}" -eq "${#SEEDS[@]}" ] || die "GPU 수와 시드 수가 다르다"

STAMP=$(date +%m%d_%H%M%S)
MANIFEST="$ROOT/outputs/${TAG}_${STAMP}_manifest.json"
echo "{" > "$MANIFEST"
printf '  "tag": "%s", "stamp": "%s", "epochs": %s, "batch": %s,\n' \
  "$TAG" "$STAMP" "$EPOCHS" "$BATCH" >> "$MANIFEST"
printf '  "dataset": "%s", "task": "%s",\n  "runs": [\n' "$DATASET" "$TASK" >> "$MANIFEST"

pids=()
for i in "${!SEEDS[@]}"; do
  s="${SEEDS[$i]}"; g="${GPUS[$i]}"
  out="$ROOT/outputs/${TAG}_s${s}"
  [ -e "$out" ] && die "출력이 이미 있다: $out (덮어쓰지 않는다)"
  log="$ROOT/outputs/${TAG}_s${s}.log"
  cd "$ROOT"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$g" nohup "$PY" -m umi_adapter.train \
    --dataset "$DATASET" --output "$out" --epochs "$EPOCHS" --batch "$BATCH" \
    --workers "$WORKERS" --seed "$s" --task "$TASK" > "$log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "  띄움  seed=$s  gpu=$g  pid=$pid  log=$log"
  sep=','; [ "$i" -eq $(( ${#SEEDS[@]} - 1 )) ] && sep=''
  printf '    {"seed": %s, "gpu": %s, "pid": %s, "output": "%s", "log": "%s"}%s\n' \
    "$s" "$g" "$pid" "$out" "$log" "$sep" >> "$MANIFEST"
  sleep 3
done
printf '  ]\n}\n' >> "$MANIFEST"

echo
echo "3개 띄웠다. manifest: $MANIFEST"
echo "진행 확인:"
echo "  grep -h 'Official training finished' $ROOT/outputs/${TAG}_s*.log | wc -l   # 3 이면 끝"
echo "  tail -2 $ROOT/outputs/${TAG}_s0.log"
echo "  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
