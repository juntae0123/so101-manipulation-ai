#!/usr/bin/env bash
# Collect at three object-jitter levels, measure, then queue training.
# 물체 지터 3수준 수집 → 계측 → 학습 큐 투입.
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=~/envs/aiot_v100/bin/python
export PYTHONPATH="$PWD" MUJOCO_GL=egl
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

names=(j0 j25 j50); jits=(0.0 0.025 0.05); seeds=(7000 7100 7200)

echo "### 1. 스모크 2편 — 인자·경로 확인"
for i in 0 1 2; do
  $PY tools/collect_sim.py --episodes 2 --jitter "${jits[$i]}" --seed "${seeds[$i]}" \
    --skill-id pick_place --out "datasets/_smoke_${names[$i]}" --author 김준태
done
echo "### 스모크 OK"

echo "### 2. 본수집 60편 x 3"
for i in 0 1 2; do
  t0=$(date +%s)
  $PY tools/collect_sim.py --episodes 60 --jitter "${jits[$i]}" --seed "${seeds[$i]}" \
    --skill-id pick_place --out "datasets/sim_pick_${names[$i]}" --author 김준태 --log
  echo "### ${names[$i]} 수집 $(( $(date +%s) - t0 ))초"
done

echo "### 3. 외삽 설명력"
$PY tools/probe_action_structure.py --data datasets/sim_pick_j0  --out out/as_j0.json
$PY tools/probe_action_structure.py --data datasets/sim_pick_j25 --out out/as_j25.json
$PY tools/probe_action_structure.py --data datasets/sim_pick_j50 --out out/as_j50.json

echo "### 4. 학습 큐 작성"
mkdir -p queue/pending
for n in j0 j25 j50; do
  cat > "queue/pending/jitter_${n}.yaml" <<YAML
name: jitter_${n}
prereg: docs/PREREG_jitter_campaign_0914.md
kind: repeat_runs
data: datasets/sim_pick_${n}
cameras: cam_wrist
runs: 3
epochs: 30
episodes: 100
seed_base: 0
eval_seed_base: 3000
action_space: joint_delta_gripper_binary
device: cuda
policy_device: cpu
YAML
done
ls -l queue/pending/

echo "### 5. 큐 검증"
$PY tools/run_queue.py --dry-run --allow-dirty

echo "### 6. 큐 실행"
$PY tools/run_queue.py --parallel 2 --allow-dirty

echo "### 캠페인 종료"
