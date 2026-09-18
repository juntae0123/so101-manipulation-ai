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

# ── 0. 관통 스모크 ─────────────────────────────────────────────────────────
# 학습 -> 체크포인트 -> 롤아웃 -> 성공률 파싱까지 **실제 호출 사슬**을 1회 통과한다.
# 단위 테스트로는 이음매가 안 잡힌다. 2026-09-15 에 ACTPolicy.name 을 "act" 로 바꿨다가
# rollout 의 게이트 키(`success_rates["bc"]`)가 끊겨 8잡이 전부 죽었다. 이 단계가
# 있었으면 2분에 잡혔다. 캠페인 40~60분을 버리기 전에 여기서 멈춘다.
echo "### 0. 관통 스모크 (chunk=8, 1시드 x 2편)"
$PY tools/repeat_runs.py \
  --data datasets/sim_pick_cmd --target-sidecar command --tag _smoke_chunk8 --chunk 8 \
  --runs 1 --epochs 1 --episodes 2 --seed-base 900 --eval-seed-base 9000 \
  --action-space joint_delta_gripper_binary --cameras cam_wrist \
  --device cuda --policy-device cpu && rc=0 || rc=$?
# ⚠️ rc=1 은 **배포 게이트 실패**다. 실행 오류가 아니다 (run_queue.py 와 같은 규약).
# 1epoch·2편짜리 스모크가 게이트 20% 를 넘을 리 없다. 여기서 보는 것은 성능이 아니라
# **호출 사슬이 끝까지 도는가** 하나다.
if [ "$rc" -gt 1 ]; then
  echo "✗ 스모크 실패 (rc=$rc). 캠페인을 돌리지 않는다."
  exit 1
fi
echo "### 스모크 통과 (rc=$rc) — 호출 사슬이 끝까지 돈다"

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
# ⚠️ run_queue 는 로그를 out/queue_<시각>/ 에 쓴다. out/logs 를 보면 큐 결과가
# 하나도 안 잡힌다 (2026-09-15 에 두 번 걸렸다). 가장 최근 큐 디렉터리를 쓴다.
LATEST_QUEUE=$(ls -td out/queue_* 2>/dev/null | head -1)
echo "큐 로그: $LATEST_QUEUE"
$PY tools/pool_rollouts.py "$LATEST_QUEUE"
echo "### 끝. out/ANALYSIS_latest.md 를 보라"
