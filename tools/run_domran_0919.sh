#!/usr/bin/env bash
# 카메라 pose 도메인 랜덤화 재학습 — 생성 → 병합 → export → 학습 → 평가
#
# 사전등록: AI/docs/PREREG_domain_randomization_0919.md (실행 전 작성)
# 왜: MEASURE_camera_mount_sweep_0919 — 현 정책이 카메라 ±2도에서 무너진다.
#     줄 사양이 없어서 마운트 정밀도가 아니라 학습으로 푼다 (D-AI-73).
#
# 인자 대조 🟢 (소스 확인, 2026-09-19)
#   simulation/generate.py  --output --episodes --seed --overview --object --task
#   umi_adapter/export.py   --source --session --dataset
#   umi_adapter/train.py    --dataset --output --epochs --steps --batch --resume
#                           --init-checkpoint --workers --lr --task  (+패치: --seed --override)
#   simulation/evaluate.py  --checkpoint --output --seed --episodes --seconds
#                           --action-steps --grip-preload --task --image-perturb(-sigma)
#   환경: MUJOCO_GL=egl 필수. 없으면 DISPLAY 없어서 렌더러가 즉사한다
#   env : handoff312 (MuJoCo·학습). VLM 만 aiot_v100 이다
#
# 실행: nohup bash ~/S15P21A103/AI/tools/run_domran_0919.sh > ~/handoff/outputs/domran.log 2>&1 &
# 완료: grep -c DOMRAN0919_DONE ~/handoff/outputs/domran.log
#
# 되돌리기: configs/_domran/ 와 outputs/domran_* 를 지우면 끝이다.

set -euo pipefail
export MUJOCO_GL=egl

R=$HOME/handoff
PY=$HOME/envs/handoff312/bin/python
TOOLS=$HOME/S15P21A103/AI/tools
TAG=domran
GROUPS=${GROUPS:-10}
PER=${PER:-10}
SEED0=${SEED0:-6000}
EPOCHS=${EPOCHS:-60}
BATCH=${BATCH:-8}
WORKERS=${WORKERS:-4}
GPU=${GPU:-1}
OBJECT=${OBJECT:-cylinder}
EVAL_SEED=${EVAL_SEED:-7000}
EVAL_N=${EVAL_N:-20}

cd "$R"

# ⚠️ 2026-09-19 — 셸에 GROUPS 가 이미 있어서 `${GROUPS:-10}` 이 100 을 집었고,
#    100편이 아니라 1000편을 생성했다. **해석된 값을 안 찍은 것이 원인**이다.
#    "모수를 같이 찍는다" 를 스크립트에도 적용한다.
LOCK=outputs/.${TAG}.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "!! 이미 돌고 있다 (pid $(cat "$LOCK")). 중복 실행하면 같은 디렉터리를 서로 지운다"
  exit 9
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "== 0. 해석된 설정 =="
printf '  %-10s %s\n' GROUPS "$GROUPS" PER "$PER" SEED0 "$SEED0" EPOCHS "$EPOCHS" \
       BATCH "$BATCH" WORKERS "$WORKERS" GPU "$GPU" OBJECT "$OBJECT" \
       EVAL_SEED "$EVAL_SEED" EVAL_N "$EVAL_N"
echo "  총 생성 편수 $((GROUPS * PER)) · 학습 epoch $EPOCHS"
if [ "$((GROUPS * PER))" -gt 200 ] && [ "${ALLOW_BIG:-0}" != "1" ]; then
  echo "!! 총 편수 $((GROUPS * PER)) 는 사전등록(100편)보다 크다."
  echo "   셸에 GROUPS/PER 가 남아 있지 않은지 확인하라: GROUPS=[$GROUPS] PER=[$PER]"
  echo "   의도한 것이면 ALLOW_BIG=1 을 붙여라."
  exit 8
fi

echo
echo "== 0-1. 착수 전 검사 =="
for f in configs/can_side.yaml configs/camera.yaml simulation/generate.py \
         umi_adapter/export.py umi_adapter/train.py simulation/evaluate.py; do
  [ -e "$f" ] || { echo "!! 없다: $f"; exit 2; }
done
[ -x "$PY" ] || { echo "!! python 없다: $PY"; exit 2; }
u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 99999)
echo "GPU $GPU 사용중 ${u} MiB"

echo
echo "== 1. 정답 아는 행 — 공칭 카메라로 2편 생성해 리프트 대조 =="
# 기록값: 100편 seed 6000-6099 리프트 0.1193 +- 0.0001 m
CTRL=outputs/${TAG}_control
rm -rf "$CTRL"
$PY simulation/generate.py --output "$CTRL" --episodes 2 --seed "$SEED0" \
    --object "$OBJECT" --task configs/can_side.yaml > outputs/${TAG}_control.log 2>&1
$PY - "$CTRL" <<'PYEOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
eps = sorted(d.glob("episode_*"))
lifts = []
for e in eps:
    m = json.loads((e / "metadata.json").read_text())
    for k in ("max_lift_m", "lift_m", "lift"):
        if k in m:
            lifts.append(float(m[k])); break
print(f"생성 {len(eps)} / 기대 2 · 리프트 {lifts}")
if len(eps) != 2:
    raise SystemExit("!! 2편이 안 나왔다. 물체 종류·설정이 다르다. 중단")
if not lifts:
    raise SystemExit("!! metadata 에서 리프트 키를 못 찾았다. 키 이름을 확인하라: "
                     + str(sorted(json.loads((eps[0]/'metadata.json').read_text()))))
bad = [x for x in lifts if abs(x - 0.1193) > 0.0005]
if bad:
    raise SystemExit(f"!! 리프트가 기록값 0.1193±0.0005 와 다르다: {bad}. "
                     "기준선과 같은 조건이 아니다. 중단")
print("정답 아는 행 통과")
PYEOF

echo
echo "== 2. 카메라 설정 ${GROUPS}종 생성 =="
$PY "$TOOLS/make_domran_cameras.py" --handoff "$R" --groups "$GROUPS" \
    --episodes-per-group "$PER" --seed0 "$SEED0"

echo
echo "== 3. 그룹별 시연 생성 =="
MERGED=outputs/${TAG}_demos
rm -rf "$MERGED"; mkdir -p "$MERGED"
for i in $(seq 0 $((GROUPS - 1))); do
  g=g$i
  s=$((SEED0 + i * PER))
  out=outputs/${TAG}_raw_${g}
  rm -rf "$out"
  echo "-- $g  seed ${s}~$((s + PER - 1))"
  $PY simulation/generate.py --output "$out" --episodes "$PER" --seed "$s" \
      --object "$OBJECT" --task "configs/_domran/task_${g}.yaml"
  for ep in "$out"/episode_*; do cp -r "$ep" "$MERGED/"; done
done
n=$(ls -d "$MERGED"/episode_* 2>/dev/null | wc -l)
echo "병합 편수 $n / 기대 $((GROUPS * PER))"
[ "$n" -eq $((GROUPS * PER)) ] || { echo "!! 편수가 안 맞는다. 중단"; exit 3; }

echo
echo "== 4. zarr export =="
DS=outputs/ds_${TAG}.zarr.zip
rm -f "$DS"
$PY umi_adapter/export.py --source "$MERGED" --session outputs/${TAG}_session --dataset "$DS"
$PY umi_adapter/check_dataset.py "$DS"

echo
echo "== 5. 학습 (${EPOCHS} epoch, GPU ${GPU}) =="
OUT=outputs/${TAG}_s0
rm -rf "$OUT"
CUDA_VISIBLE_DEVICES=$GPU $PY -m umi_adapter.train --dataset "$DS" --output "$OUT" \
    --epochs "$EPOCHS" --batch "$BATCH" --workers "$WORKERS" --task configs/can_side.yaml

CKPT=$OUT/checkpoints/latest.ckpt
[ -e "$CKPT" ] || { echo "!! 체크포인트가 없다: $CKPT"; exit 4; }

echo
echo "== 6. 평가 — 공칭 + 강건성 4조건 + 판별 1조건 =="
# 스윕이 만들어둔 교란 설정을 그대로 쓴다. 없으면 먼저 --plan 을 돌려야 한다.
for c in a0 "a+5" "a-5" "z+10" "z-10" DISC; do
  t=configs/_sweep_camera/task_${c}.yaml
  [ -e "$t" ] || { echo "!! $t 가 없다. sweep_camera_mount.py --plan 을 먼저 돌려라"; exit 5; }
  echo "-- 평가 $c"
  CUDA_VISIBLE_DEVICES=$GPU $PY simulation/evaluate.py --checkpoint "$CKPT" \
      --task "$t" --output outputs/${TAG}_eval/${c} --seed "$EVAL_SEED" \
      --episodes "$EVAL_N" --action-steps 4 > outputs/${TAG}_eval_${c}.log 2>&1 || true
done

echo
echo "== 7. 집계 =="
$PY - <<'PYEOF'
import json, pathlib
root = pathlib.Path.home() / "handoff/outputs/domran_eval"
conds = ["a0", "a+5", "a-5", "z+10", "z-10", "DISC"]
res = {}
for c in conds:
    f = root / c / "evaluation.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    n = len(d["episodes"]); k = sum(bool(e["success"]) for e in d["episodes"])
    res[c] = (k, n)
print(f"집계 {len(res)} / 조건 {len(conds)}")
for c in conds:
    if c in res:
        k, n = res[c]
        print(f"  {c:<5} {k:>2}/{n}  = {k/n*100:5.1f}%")
    else:
        print(f"  {c:<5} 결과 없음")
if "a0" not in res:
    raise SystemExit("!! 공칭(a0) 결과가 없다. 게이트를 판정하지 않는다")
g1 = res["a0"][0] / res["a0"][1] >= 0.87
rob = [c for c in ("a+5", "a-5", "z+10", "z-10") if c in res]
if len(rob) < 4:
    raise SystemExit(f"!! 강건성 조건이 {len(rob)}/4 뿐이다. G2 를 판정하지 않는다")
avg = sum(res[c][0] / res[c][1] for c in rob) / len(rob)
g2 = avg >= 0.70
g3 = ("DISC" in res) and (res["DISC"][0] / res["DISC"][1] < 0.50)
print(f"\nG1 공칭 >= 87%    {'통과' if g1 else '미달'}")
print(f"G2 강건 평균 >= 70%  {avg*100:.1f}%  {'통과' if g2 else '미달'}  "
      f"(현 정책 평균 7.5%)")
print(f"G3 판별 45도 < 50%  {'통과' if g3 else '미달 — 카메라가 안 읽히는 것일 수 있다'}")
print(f"\n판정: {'채택' if (g1 and g2 and g3) else '미채택'}")
if not g2:
    print("  ⚠️ 게이트를 낮추지 않는다. 범위(±5도/±10mm)를 바꿔 재실행한다")
PYEOF

echo
echo DOMRAN0919_DONE
