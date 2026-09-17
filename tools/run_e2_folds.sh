#!/usr/bin/env bash
# E2 분할 5개로 n 확대 (D-AI-59 사전등록)
#
# 왜: E2 결론(시뮬 사전학습이 돕는다)의 유일한 약점이 n 이다. 홀드아웃 13편.
#     게이트를 옮기지 않고 n 을 올린다.
#
# 설계: 분할 시드 42~46 로 각각 train/holdout 재변환 → fold 마다 A·B 1회씩 학습.
#       B 의 출발 체크포인트는 기존 e2_C 하나로 고정해 C 시드를 변수에서 뺀다.
#       학습 시드는 0 고정 (시드 간 변동 1.52mm = 3.9% 로 이미 작다고 측정됨).
#
# 정답을 아는 행: 시드 42 는 기존 분할과 같아야 한다.
#   기존  train kept 56 · frames 1021   /   holdout kept 13 · frames 226
#   재현 안 되면 --task 등 조건이 다른 것이므로 **한 fold 도 학습하지 않고 멈춘다.**
#
# 실행:  nohup bash ~/S15P21A103/AI/tools/run_e2_folds.sh > ~/handoff/outputs/e2folds.log 2>&1 &
# 완료:  grep -c E2FOLDS_DONE_20260918 ~/handoff/outputs/e2folds.log

set -uo pipefail
R=$HOME/handoff
PY=$HOME/envs/handoff312/bin/python
TOOLS=$HOME/S15P21A103/AI/tools
SRC=/home/j-j15a103/S15P21A103_umi/AI/datasets/umi_real_relative_20260911_v10
TASK=${TASK:-configs/can_side.yaml}
CKPT_C=outputs/e2_C/checkpoints/latest.ckpt
SEEDS=(42 43 44 45 46)
GPUS=(1 2 3 4 6)
cd "$R" || exit 2

echo "== 착수 전 검사 =="
for f in "$SRC/dataset.json" "$TASK" "$CKPT_C"; do
  [ -e "$f" ] || { echo "!! 없다: $f"; exit 2; }
done
grep -q -- "--override" umi_adapter/train.py || { echo "!! --override 패치 없음"; exit 2; }
for g in "${GPUS[@]}"; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  [ "$u" -lt 500 ] || { echo "!! GPU $g 가 ${u}MiB 사용 중"; exit 2; }
done
echo "   OK · task=$TASK"

chk () {   # chk <provenance> <기대 편수> <기대 프레임>
  $PY - "$1" "$2" "$3" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
k, f = int(d["episodes_kept"]), int(d["frames"])
ek, ef = int(sys.argv[2]), int(sys.argv[3])
print(f"   {sys.argv[1]}  편 {k}/{ek}  프레임 {f}/{ef}", flush=True)
sys.exit(0 if (k == ek and f == ef) else 1)
PY
}

echo
echo "== 1. 시드 42 재현 검정 =="
for only in train holdout; do
  $PY "$TOOLS/convert_v10_to_umi.py" --dataset "$SRC" --task "$TASK" \
      --out "outputs/ds_f42_$only" --split-seed 42 --holdout 14 --only "$only" \
      > "outputs/cv_f42_$only.log" 2>&1 \
    || { echo "!! 변환 실패 ($only). outputs/cv_f42_$only.log 확인"; exit 3; }
done
chk outputs/ds_f42_train.provenance.json   56 1021 || { echo "!! 재현 실패 — 조건이 다르다. 중단"; exit 3; }
chk outputs/ds_f42_holdout.provenance.json 13  226 || { echo "!! 재현 실패 — 조건이 다르다. 중단"; exit 3; }
echo "   재현 OK — task 조건이 기존과 같다"

echo
echo "== 2. 나머지 분할 변환 (43~46) =="
for s in 43 44 45 46; do
  for only in train holdout; do
    $PY "$TOOLS/convert_v10_to_umi.py" --dataset "$SRC" --task "$TASK" \
        --out "outputs/ds_f${s}_$only" --split-seed "$s" --holdout 14 --only "$only" \
        > "outputs/cv_f${s}_$only.log" 2>&1 \
      || { echo "!! 변환 실패 (seed $s / $only)"; exit 3; }
    n=$($PY -c "import json;print(json.load(open('outputs/ds_f${s}_$only.provenance.json'))['episodes_kept'])")
    echo "   seed $s $only  편 $n"
  done
done

train_batch () {   # train_batch <조건 A|B>
  local cond=$1 i=0 pids=()
  for s in "${SEEDS[@]}"; do
    local g=${GPUS[$i]} out="outputs/e2f_${cond}_$s" extra=""
    [ "$cond" = "B" ] && extra="--init-checkpoint $CKPT_C"
    if [ -e "$out" ]; then echo "   건너뜀 (이미 있음): $out"; i=$((i+1)); continue; fi
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$g" nohup $PY -m umi_adapter.train \
      --dataset "outputs/ds_f${s}_train.zarr.zip" --output "$out" \
      --epochs 60 --batch 8 --workers 4 --seed 0 --task "$TASK" $extra \
      --override task.action_horizon=8 --override task.obs_down_sample_steps=1 \
      > "outputs/e2f_${cond}_$s.log" 2>&1 &
    pids+=($!); echo "   $cond seed $s → GPU $g  pid $!"
    i=$((i+1))
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "   $cond 배치 완료"
}

echo; echo "== 3. A조건 5 fold 학습 =="; train_batch A
echo; echo "== 4. B조건 5 fold 학습 =="; train_batch B

echo
echo "== 5. 평가 =="
for s in "${SEEDS[@]}"; do
  for cond in A B; do
    CUDA_VISIBLE_DEVICES=1 $PY "$TOOLS/eval_holdout_error.py" \
      --checkpoint "outputs/e2f_${cond}_$s/checkpoints/latest.ckpt" \
      --dataset "outputs/ds_f${s}_holdout.zarr.zip" \
      --out "outputs/err_f${s}_$cond.json" 2>&1 | tail -4
  done
done

echo
echo "== 6. fold 별 B-A paired =="
for s in "${SEEDS[@]}"; do
  echo "--- fold $s ---"
  $PY "$TOOLS/eval_holdout_error.py" --compare "outputs/err_f${s}_A.json" "outputs/err_f${s}_B.json"
done

echo
echo "E2FOLDS_DONE_20260918"
