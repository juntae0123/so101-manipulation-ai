#!/usr/bin/env bash
# 0918 실 시연 74편(축 수정본) 5 fold — A/B 조건 대조
#
# 왜: 2026-09-18 단일 fold 에서 trans 개선 69.2% · rot 44.2% 가 나왔다.
#     fold 하나로는 판정할 수 없다. 게이트를 옮기지 말고 n 을 올린다.
#     E2(0911)와 같은 설계라 바로 비교된다.
#
# 데이터: umi_real_relative_20260918_v10_orbslam_cadtcp_video_aligned_provisional_74ep
#         (황도경 · SHA 3b355a7a… · camera→TCP 축 수정본, jaw 축 오차 89.87° → 0.11°)
#
# 정답을 아는 행: 분할시드 42 는 오늘 밤 측정값과 같아야 한다.
#   train 편 60 · 프레임 3898   /   holdout 편 14 · 프레임 816
#   재현 안 되면 조건이 다른 것이므로 **한 fold 도 학습하지 않고 멈춘다.**
#
# 실행: nohup bash ~/S15P21A103/AI/tools/run_folds_0918.sh > ~/handoff/outputs/folds0918.log 2>&1 &
# 완료: grep -c FOLDS0918_DONE ~/handoff/outputs/folds0918.log

set -uo pipefail
R=$HOME/handoff
PY=$HOME/envs/handoff312/bin/python
TOOLS=$HOME/S15P21A103/AI/tools
SRC=$HOME/S15P21A103_umi/AI/datasets/umi_real_relative_20260918_v10_orbslam_cadtcp_video_aligned_provisional_74ep
TASK=${TASK:-configs/can_side.yaml}
CKPT_C=outputs/e2_C/checkpoints/latest.ckpt
TAG=f0918
SEEDS=(42 43 44 45 46)
GPUS=(1 2 3 4 6)
HOLDOUT=14
EPOCHS=60
MIN_FREE_GB=40
cd "$R" || exit 2

echo "== 착수 전 검사 =="
for f in "$SRC/dataset.json" "$TASK" "$CKPT_C"; do
  [ -e "$f" ] || { echo "!! 없다: $f"; exit 2; }
done
grep -q -- "--override" umi_adapter/train.py || { echo "!! --override 패치 없음"; exit 2; }
grep -q -- "--init-checkpoint" umi_adapter/train.py || { echo "!! --init-checkpoint 없음"; exit 2; }
for g in "${GPUS[@]}"; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null) \
    || { echo "!! GPU $g 조회 실패"; exit 2; }
  [ -n "$u" ] && [ "$u" -lt 500 ] || { echo "!! GPU $g 가 ${u}MiB 사용 중. 남의 잡 위에 안 얹는다"; exit 2; }
done
free_gb=$(df -BG --output=avail "$R" | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge "$MIN_FREE_GB" ] || { echo "!! 디스크 ${free_gb}GB 남음. ${MIN_FREE_GB}GB 필요 (zarr 10개 + ckpt 10개)"; exit 2; }
echo "   OK · task=$TASK · 디스크 ${free_gb}GB · GPU ${GPUS[*]}"

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
echo "== 1. 분할시드 42 재현 검정 (정답 아는 행) =="
for only in train holdout; do
  out="outputs/ds_${TAG}_42_$only"
  [ -e "$out.zarr.zip" ] && { echo "   이미 있음, 건너뜀: $out"; continue; }
  $PY "$TOOLS/convert_v10_to_umi.py" --dataset "$SRC" --task "$TASK" \
      --out "$out" --split-seed 42 --holdout "$HOLDOUT" --only "$only" \
      > "outputs/cv_${TAG}_42_$only.log" 2>&1 \
    || { echo "!! 변환 실패 ($only)"; exit 3; }
done
chk "outputs/ds_${TAG}_42_train.provenance.json"   60 3898 || { echo "!! 재현 실패 — 조건이 다르다. 중단"; exit 3; }
chk "outputs/ds_${TAG}_42_holdout.provenance.json" 14  816 || { echo "!! 재현 실패 — 조건이 다르다. 중단"; exit 3; }
echo "   재현 OK"

echo
echo "== 2. 나머지 분할 변환 (43~46) =="
for s in 43 44 45 46; do
  for only in train holdout; do
    out="outputs/ds_${TAG}_${s}_$only"
    [ -e "$out.zarr.zip" ] && { echo "   이미 있음: $out"; continue; }
    $PY "$TOOLS/convert_v10_to_umi.py" --dataset "$SRC" --task "$TASK" \
        --out "$out" --split-seed "$s" --holdout "$HOLDOUT" --only "$only" \
        > "outputs/cv_${TAG}_${s}_$only.log" 2>&1 \
      || { echo "!! 변환 실패 (seed $s / $only)"; exit 3; }
  done
  a=$($PY -c "import json;print(json.load(open('outputs/ds_${TAG}_${s}_train.provenance.json'))['episodes_kept'])")
  b=$($PY -c "import json;print(json.load(open('outputs/ds_${TAG}_${s}_holdout.provenance.json'))['episodes_kept'])")
  echo "   seed $s  train $a  holdout $b  합 $((a+b))"
done

train_batch () {   # train_batch <A|B>
  local cond=$1 i=0 pids=()
  for s in "${SEEDS[@]}"; do
    local g=${GPUS[$i]} out="outputs/${TAG}_${cond}_$s" extra=""
    [ "$cond" = "B" ] && extra="--init-checkpoint $CKPT_C"
    if [ -e "$out" ]; then echo "   건너뜀 (이미 있음): $out"; i=$((i+1)); continue; fi
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$g" nohup $PY -m umi_adapter.train \
      --dataset "outputs/ds_${TAG}_${s}_train.zarr.zip" --output "$out" \
      --epochs "$EPOCHS" --batch 8 --workers 4 --seed 0 --task "$TASK" $extra \
      --override task.action_horizon=8 --override task.obs_down_sample_steps=1 \
      > "outputs/${TAG}_${cond}_$s.log" 2>&1 &
    pids+=($!); echo "   $cond seed $s → GPU $g  pid $!"
    i=$((i+1)); sleep 3
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "   $cond 배치 완료"
}

echo; echo "== 3. A조건 (warm start 없음) 5 fold =="; train_batch A
echo; echo "== 4. B조건 (시뮬 사전학습에서 출발) 5 fold =="; train_batch B

echo
echo "== 5. 홀드아웃 평가 =="
for s in "${SEEDS[@]}"; do
  for cond in A B; do
    CUDA_VISIBLE_DEVICES=1 $PY "$TOOLS/eval_holdout_error.py" \
      --checkpoint "outputs/${TAG}_${cond}_$s/checkpoints/latest.ckpt" \
      --dataset "outputs/ds_${TAG}_${s}_holdout.zarr.zip" \
      --out "outputs/err_${TAG}_${s}_$cond.json" 2>&1 | tail -4
  done
done

echo
echo "== 6. fold 별 B−A paired =="
for s in "${SEEDS[@]}"; do
  echo "--- fold $s ---"
  $PY "$TOOLS/eval_holdout_error.py" --compare \
     "outputs/err_${TAG}_${s}_A.json" "outputs/err_${TAG}_${s}_B.json"
done

echo
echo "FOLDS0918_DONE"
