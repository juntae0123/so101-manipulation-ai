#!/usr/bin/env bash
# Collect DAgger segments with the command sidecar, then train on them.
# command 사이드카를 붙여 DAgger 세그먼트를 수집하고, 그 데이터로 학습한다.
#
# 왜 스크립트인가: 한 줄 명령을 매번 손으로 조립하다가 PYTHONPATH·MUJOCO_GL 을
# 반복해서 빠뜨렸다. 환경은 여기 박아두고 호출은 한 줄로 끝낸다.
#
#   # [서버]
#   nohup bash AI/tools/run_dagger_cmd.sh > AI/out/logs/dagger.log 2>&1 &

set -eu
cd ~/S15P21A103/AI
PY=~/envs/aiot_v100/bin/python
export PYTHONPATH="$PWD"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
mkdir -p out/logs

BEHAVIOR=checkpoints/bc/sim_pick_cmd_chunk8_cmd_seed2.pt
OUT=datasets/dagger_cmd

echo "### 0. 행동 정책 확인"
test -f "$BEHAVIOR" || { echo "✗ 체크포인트가 없다: $BEHAVIOR"; exit 1; }

echo "### 1. DAgger 수집 60편"
rm -rf "$OUT"
$PY tools/collect_dagger_segments.py \
  --policy-ckpt "$BEHAVIOR" --episodes 60 --seed-base 4000 --out "$OUT" --log

echo "### 2. 사이드카 검증 — npz 개수와 같아야 한다"
N_NPZ=$(ls "$OUT"/*.npz 2>/dev/null | wc -l)
N_CMD=$(ls "$OUT"/*.command.npy 2>/dev/null | wc -l)
echo "npz $N_NPZ · command $N_CMD"
if [ "$N_NPZ" -eq 0 ] || [ "$N_NPZ" -ne "$N_CMD" ]; then
  echo "✗ 사이드카가 npz 와 개수가 다르다. 학습하지 않는다."
  exit 1
fi

echo "### 3. 학습 3시드 x 롤아웃 100편 — command 타깃, chunk 8"
$PY tools/repeat_runs.py \
  --data "$OUT" --target-sidecar command --tag dagger_cmd_chunk8_v2 --chunk 8 \
  --runs 3 --epochs 30 --episodes 100 --seed-base 0 --eval-seed-base 3000 \
  --action-space joint_delta_gripper_binary --cameras cam_wrist \
  --device cuda --policy-device cpu --log && rc=0 || rc=$?
# rc=1 은 배포 게이트 실패다. 결과이지 오류가 아니다.
[ "$rc" -gt 1 ] && { echo "✗ 학습/평가 실패 (rc=$rc)"; exit 1; }

echo "### 4. 대조군 재측정 — 표준화 하한 수정(1e-8 -> RANGE_TOLERANCE)이 여기도 영향을 준다"
# 같은 코드로 재지 않으면 DAgger 와 비교가 성립하지 않는다. 0915 에 이미 한 번
# 겪었다 — grip_mid 수정 뒤 대조군이 9.3% 에서 8.0% 로 움직였다.
$PY tools/repeat_runs.py \
  --data datasets/sim_pick_cmd --target-sidecar command --tag ctrl_chunk8_v2 --chunk 8 \
  --runs 3 --epochs 30 --episodes 100 --seed-base 0 --eval-seed-base 3000 \
  --action-space joint_delta_gripper_binary --cameras cam_wrist \
  --device cuda --policy-device cpu --log && rc=0 || rc=$?
[ "$rc" -gt 1 ] && { echo "✗ 대조군 재측정 실패 (rc=$rc)"; exit 1; }

echo "### 5. 합산"
$PY tools/pool_rollouts.py out/logs
echo "### 끝. out/ANALYSIS_latest.md"
