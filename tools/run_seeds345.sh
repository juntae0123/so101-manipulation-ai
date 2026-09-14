#!/usr/bin/env bash
# Seeds 3-5 for the image-ablation comparison, then pool every log.
# 이미지 어블레이션 비교의 시드 3~5, 그 뒤 전체 로그 합산.
#
# 왜 대조군에도 시드를 더 넣나: 2026-09-14 의 cmd_noise96 에서 시드 2 의 val 이
# 자명한 예측기(2.55132) 수준으로 나왔다. 노이즈 효과가 아니라 학습 실패다.
# 노이즈 쪽만 늘리면 그 붕괴가 노이즈 탓인지 알 수 없다.
#
# repeat_runs 는 단일 인스턴스 락을 쓴다. 순차로만 돈다.
#
#   # [서버]
#   nohup bash AI/tools/run_seeds345.sh > AI/out/logs/seeds345.log 2>&1 &

set -x
cd ~/S15P21A103/AI
PY=~/envs/aiot_v100/bin/python
export PYTHONPATH="$PWD"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
mkdir -p out/logs

COMMON="--data datasets/sim_pick_cmd --target-sidecar command"
COMMON="$COMMON --runs 3 --epochs 30 --episodes 100"
COMMON="$COMMON --seed-base 3 --eval-seed-base 3000"
COMMON="$COMMON --action-space joint_delta_gripper_binary --cameras cam_wrist"
COMMON="$COMMON --device cuda --policy-device cpu --log"

$PY tools/repeat_runs.py $COMMON --tag cmd_noise96_s345 --image-noise 96
$PY tools/repeat_runs.py $COMMON --tag cmd_s345

$PY tools/pool_rollouts.py out/logs
echo "### 끝. out/ANALYSIS_latest.md 를 보라"
