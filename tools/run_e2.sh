#!/usr/bin/env bash
# E2 3팔 — 트랙 A 가 IK 게이트 재정의에 동의한 뒤에만 돌린다.
# 실행:  bash ~/run_e2.sh
set -uo pipefail
R=$HOME/handoff; PY=$HOME/envs/handoff312/bin/python
cd $R

echo "== 착수 전 검사 =="
for f in outputs/ds_6000.zarr.zip configs/can_side.yaml; do
  [ -e "$f" ] || { echo "!! 없다: $f"; exit 2; }
done
grep -q -- "--override" umi_adapter/train.py || { echo "!! --override 패치가 없다"; exit 2; }
for g in 1 2 3; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$u" -lt 500 ] || { echo "!! GPU $g 가 ${u}MiB 쓰는 중"; exit 2; }
done
echo "   OK"

echo
echo "== C팔: 시뮬 전용 사전학습 (H8) — GPU 1 =="
[ -e outputs/e2_C ] && { echo "!! outputs/e2_C 가 이미 있다. 덮어쓰지 않는다"; exit 3; }
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 nohup $PY -m umi_adapter.train \
  --dataset outputs/ds_6000.zarr.zip --output outputs/e2_C \
  --epochs 60 --batch 8 --workers 4 --seed 0 \
  --override task.action_horizon=8 --task configs/can_side.yaml \
  > outputs/e2_C.log 2>&1 &
echo "   pid $!  로그 outputs/e2_C.log"
echo
echo "A팔·B팔은 실데이터 변환본이 필요하다 (convert_real_umi.py 확장 대기)."
echo "C팔이 끝나면 그 체크포인트가 B팔의 --init-checkpoint 가 된다."
