#!/usr/bin/env bash
# Eight chunk conditions, queued and run in parallel across the allocated GPUs.
# 청크 조건 8개를 큐에 넣고 할당 GPU 에 병렬로 돌린다.
#
# 사전등록: docs/PREREG_chunking_0915.md (10-보충 절에 8조건과 예측을 실행 전에 박았다)
#
# 자원 실측 2026-09-15 🟢 — 잡당 CPU 151% · RSS 0.86GB · GPU 808MiB.
# 서버 80코어 · 1495GB 가용 · V100 32GB x5. 병렬 8 은 장당 최대 2잡, 1.6GB/장이다.
#
# ⚠️ G0(chunk=1 이 기존 BC 와 일치)이 통과한 뒤에만 돌린다.
#
#   # [서버]
#   nohup bash AI/tools/run_chunking_campaign.sh > AI/out/logs/chunking.log 2>&1 &

set -eu
cd ~/S15P21A103/AI
PY=~/envs/aiot_v100/bin/python
export PYTHONPATH="$PWD"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
mkdir -p out/logs queue/pending

write_item () {          # $1=name  $2=chunk  $3=sidecar(또는 -)  $4=noise(또는 -)
  local f="queue/pending/$1.yaml"
  {
    echo "name: $1"
    echo "prereg: docs/PREREG_chunking_0915.md"
    echo "kind: repeat_runs"
    echo "data: datasets/sim_pick_cmd"
    echo "cameras: cam_wrist"
    echo "chunk: $2"
    echo "runs: 3"
    echo "epochs: 30"
    echo "episodes: 100"
    echo "seed_base: 0"
    echo "eval_seed_base: 3000"
    echo "action_space: joint_delta_gripper_binary"
    echo "device: cuda"
    echo "policy_device: cpu"
    [ "$3" != "-" ] && echo "target_sidecar: $3"
    [ "$4" != "-" ] && echo "image_noise: $4"
  } > "$f"
  echo "  $f"
}

echo "### 1. 큐 작성 (8조건)"
# K 스윕 — command 타깃. 대조군은 chunk=1 command 9.3% [7.3, 11.9] n=600
write_item chunk2_cmd   2 command -
write_item chunk4_cmd   4 command -
write_item chunk8_cmd   8 command -
write_item chunk16_cmd 16 command -
# v6 대리 — 계약 action[t]=state[t+1] 이므로 청크가 곧 미래 상태 궤적이다
write_item chunk8_traj   8 - -
write_item chunk16_traj 16 - -
# 이미지 어블레이션 — 청킹이 "관측 한 번 보고 가던 대로 K칸" 지름길을 만드는가
write_item chunk8_cmd_noise96  8 command 96
write_item chunk8_traj_noise96 8 -       96

echo "### 2. 큐 검증 (실행 안 함)"
$PY tools/run_queue.py --dry-run --allow-dirty

echo "### 3. 병렬 실행 — 전부 우리 카드(GPU 2)에. 8잡 x 808MiB ≈ 6.5GB / 32GB"
$PY tools/run_queue.py --parallel 8 --gpu 2 --allow-dirty

echo "### 4. 전체 로그 합산"
$PY tools/pool_rollouts.py out/logs
echo "### 끝. out/ANALYSIS_latest.md 를 보라"
