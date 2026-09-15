#!/usr/bin/env bash
# Eight conditions in parallel on GPU 2: control re-measure, DAgger alone, DAgger mixed.
# GPU 2 에 8조건 병렬: 대조군 재측정 · DAgger 단독 · DAgger 혼합.
#
# 왜 대조군을 다시 재나: 2026-09-15 에 두 가지를 고쳤다 — `grip_mid` 누락과 타깃
# 표준화 하한(1e-8 -> RANGE_TOLERANCE). 둘 다 학습 결과를 바꾼다. 옛 수치와 비교하면
# **코드가 다른 비교**가 된다. 오늘 한 번 겪었다.
#
# 왜 혼합을 넣나: DAgger 원 방식은 전문가 데이터와 교정 데이터를 섞는다. 교정만
# 쓰면 "실패 상태에서 회복하는 법"만 배운다. 0/300 을 그 반쪽 구현으로 받았다.
#
#   # [서버]
#   nohup bash AI/tools/run_dagger_campaign.sh > AI/out/logs/dcamp.log 2>&1 &

set -eu
cd ~/S15P21A103/AI
PY=~/envs/aiot_v100/bin/python
export PYTHONPATH="$PWD"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
mkdir -p out/logs queue/pending

echo "### 0. 입력 확인"
test -d datasets/sim_pick_cmd || { echo "✗ datasets/sim_pick_cmd 없음"; exit 1; }
test -d datasets/dagger_cmd   || { echo "✗ datasets/dagger_cmd 없음 — 수집부터"; exit 1; }

echo "### 1. 혼합 데이터셋 (전문가 + 교정)"
$PY tools/merge_datasets.py --src datasets/sim_pick_cmd --src datasets/dagger_cmd \
    --out datasets/mix_cmd

write_item () {          # $1=name $2=data $3=chunk $4=sidecar(-) $5=noise(-)
  local f="queue/pending/$1.yaml"
  {
    echo "name: $1"
    echo "prereg: docs/PREREG_chunking_0915.md"
    echo "kind: repeat_runs"
    echo "data: $2"
    echo "cameras: cam_wrist"
    echo "chunk: $3"
    echo "runs: 3"
    echo "epochs: 30"
    echo "episodes: 100"
    echo "seed_base: 0"
    echo "eval_seed_base: 3000"
    echo "action_space: joint_delta_gripper_binary"
    echo "device: cuda"
    echo "policy_device: cpu"
    [ "$4" != "-" ] && echo "target_sidecar: $4"
    [ "$5" != "-" ] && echo "image_noise: $5"
  } > "$f"
  echo "  $f"
}

echo "### 2. 큐 작성 (8조건)"
rm -f queue/pending/*.yaml
# 대조군 — 같은 코드로 다시 잰다
write_item v2_ctrl_c1    datasets/sim_pick_cmd  1 command -
write_item v2_ctrl_c8    datasets/sim_pick_cmd  8 command -
# DAgger 단독 — 교정 데이터만
write_item v2_dag_c1     datasets/dagger_cmd    1 command -
write_item v2_dag_c8     datasets/dagger_cmd    8 command -
# DAgger 혼합 — 정식 방식. **오늘의 핵심 조건**
write_item v2_mix_c1     datasets/mix_cmd       1 command -
write_item v2_mix_c8     datasets/mix_cmd       8 command -
# 이미지 어블레이션 — 혼합이 이미지를 쓰는가
write_item v2_mix_c8_n96 datasets/mix_cmd       8 command 96
# 계약 action 타깃 대조 (v6 대리)
write_item v2_mix_traj_c8 datasets/mix_cmd      8 -       -

echo "### 3. 큐 검증"
$PY tools/run_queue.py --dry-run --allow-dirty

echo "### 4. 병렬 실행 — 전부 GPU 2 (8잡 x 808MiB = 6.5GB / 32GB)"
$PY tools/run_queue.py --parallel 8 --gpu 2 --allow-dirty

echo "### 5. 합산"
# ⚠️ run_queue 는 로그를 out/queue_<시각>/ 에 쓴다. out/logs 를 보면 큐 결과가
# 하나도 안 잡힌다 (2026-09-15 에 두 번 걸렸다). 가장 최근 큐 디렉터리를 쓴다.
LATEST_QUEUE=$(ls -td out/queue_* 2>/dev/null | head -1)
echo "큐 로그: $LATEST_QUEUE"
$PY tools/pool_rollouts.py "$LATEST_QUEUE"
echo "### 끝. out/ANALYSIS_latest.md 와 최신 out/queue_* 디렉터리를 보라"
