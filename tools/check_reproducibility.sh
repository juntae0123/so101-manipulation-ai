#!/usr/bin/env bash
# Measure whether two identically-seeded training runs actually produce the same
# numbers. Do not claim a seeding fix works -- measure it.
# 같은 시드로 두 번 학습했을 때 정말 같은 수가 나오는지 잰다.
# 시드 수정이 먹혔다고 **주장하지 말고 측정한다.**
#
# 왜 있나 — 2026-09-16 🟢. 같은 데이터·같은 시드·같은 평가 시드로 돌린 두 캠페인의
# 시드별 성공률이 76/52/63 과 30/63/77 이었다. 그때까지 "시드를 고정했으니 재현된다"
# 고 믿고 있었다. `torch.manual_seed` 만으로는 CUDA RNG 도 cuDNN 알고리즘 선택도
# 덮이지 않는다.
#
# 짧게 돈다(epochs 5). 재현성은 마지막 val 한 자리로 충분히 갈린다 —
# 비결정이면 소수점 아래에서 이미 벌어진다.
#
#   # [서버]
#   bash tools/check_reproducibility.sh

set -u
cd ~/S15P21A103/AI || exit 1
PY=~/envs/aiot_v100/bin/python
export PYTHONPATH="$PWD"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=2
DATA=datasets/mix_cmd
mkdir -p out/logs checkpoints/bc

run () {          # $1=태그  $2=추가인자
  $PY tools/train_bc.py --data "$DATA" --seed 0 --epochs 5 --device cuda \
      --action-space joint_delta_gripper_binary --cameras cam_wrist \
      --target-sidecar command --chunk 8 \
      --out "checkpoints/bc/_repro_$1.pt" $2 \
      > "out/logs/repro_$1.log" 2>&1
  # 마지막 epoch 줄의 val 값
  grep -o 'val [0-9.]*' "out/logs/repro_$1.log" | tail -1
}

echo "=============================================="
echo " 재현성 계측 — 같은 시드로 2회씩, 두 모드"
echo " 데이터 $DATA · seed 0 · epochs 5 · GPU 2"
echo "=============================================="

echo
echo "[1/2] 기본 모드 (결정론 플래그 없음)"
A=$(run nd_a "")
B=$(run nd_b "")
echo "  1회차: $A"
echo "  2회차: $B"
[ "$A" = "$B" ] && echo "  -> 같다" || echo "  -> ⚠️ 다르다. 시드만으로는 재현되지 않는다"

echo
echo "[2/2] 결정론 모드 (--deterministic)"
C=$(run d_a "--deterministic")
D=$(run d_b "--deterministic")
echo "  1회차: $C"
echo "  2회차: $D"
[ "$C" = "$D" ] && echo "  -> 같다" || echo "  -> ⚠️ 여전히 다르다. 남은 비결정 원인이 있다"

echo
echo "읽는 법"
echo "  · [1] 다르고 [2] 같다  -> cudnn 이 원인. 재현이 필요한 실행에 --deterministic"
echo "  · [1] 다르고 [2] 다르다 -> 다른 원인이 남아 있다. DataLoader worker 시드 등"
echo "  · [1] 같다             -> 재현성은 원래 있었다. 캠페인 차이는 다른 데서 왔다"
echo
echo "⚠️ 재현되더라도 **시드 간 편차는 그대로다.** 이것은 '같은 시드가 같은 값을"
echo "   내는가' 이지 '조건을 한 시드로 판정해도 되는가' 가 아니다. 후자는 아니다."
echo
echo "정리: rm -f checkpoints/bc/_repro_*.pt"
