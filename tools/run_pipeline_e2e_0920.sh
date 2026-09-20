#!/usr/bin/env bash
# End-to-end pipeline pass on our v10 real demos.
# 우리 v10 실 시연으로 파이프라인을 끝까지 한 번 통과시킨다.
#
# 실행:
#   nohup bash ~/S15P21A103/AI/tools/run_pipeline_e2e_0920.sh > ~/S15P21A103/out/e2e_0920.log 2>&1 &
#
# 완료 감지 문자열 (다른 맥락에서 안 나오는 것):
#   "E2E_0920_FINISHED_MARKER"
#
# 원칙
#   - 단계마다 모수를 찍는다. "안 함" 과 "됐음" 이 같은 출력으로 나오지 않게 한다
#   - 한 단계가 실패해도 다음 단계가 조용히 이어지지 않는다. STATUS 에 남긴다
#   - 학습 진입점을 못 찾으면 **미실행으로 명시**한다. 통과가 아니다

set -u
PY=~/envs/handoff312/bin/python
REPO=~/S15P21A103
V10=~/S15P21A103_umi/AI/datasets/umi_real_relative_20260918_v10_orbslam_cadtcp_video_aligned_provisional_74ep
OUT=$REPO/out/e2e_0920
STAMP=$(date +%Y%m%d_%H%M%S)
EPOCHS=${EPOCHS:-60}
GPU=${GPU:-1}                      # 1·2·3·4·6 만 사용. 0·5·7·8·9 는 타인 것
mkdir -p "$OUT"
STATUS=$OUT/STATUS.txt
: > "$STATUS"

say() { echo; echo "=================== $* ==================="; }
note() { echo "[STATUS] $*" | tee -a "$STATUS"; }

say "0. 프리플라이트  $STAMP"
note "시작 $STAMP · EPOCHS=$EPOCHS · GPU=$GPU"

ok=0; tot=0
chk() { tot=$((tot+1)); if eval "$2" >/dev/null 2>&1; then ok=$((ok+1)); echo "  [있음] $1"; else echo "  [없음] $1"; fi; }
chk "python $PY"              "test -x $PY"
chk "저장소 $REPO"             "test -d $REPO"
chk "v10 데이터셋"              "test -d $V10"
chk "감사기"                   "test -f $REPO/AI/tools/audit_umi_zarr.py"
chk "변환기"                   "test -f $REPO/AI/tools/convert_v10_to_umi.py"
chk "앵커 프로브"               "test -f $REPO/AI/tools/probe_chunk_anchor.py"
echo "  프리플라이트 $ok / $tot"
note "프리플라이트 $ok / $tot"
if [ "$ok" -lt 5 ]; then note "!! 필수 경로가 없다. 중단"; echo "E2E_0920_FINISHED_MARKER"; exit 1; fi

# 가용성 확인 != 기능 확인. 실연산까지 시킨다
say "0b. GPU 실연산"
CUDA_VISIBLE_DEVICES=$GPU $PY - <<'PYEOF' 2>&1 | tee "$OUT/gpu_check.txt"
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
    a = torch.randn(512, 512, device="cuda")
    b = (a @ a).sum().item()
    print("실연산 OK  합계", round(b, 3))
else:
    print("!! CUDA 없음 — 학습 불가")
PYEOF
grep -q "실연산 OK" "$OUT/gpu_check.txt" && note "GPU 실연산 통과" || note "!! GPU 실연산 실패 — 학습 단계는 미실행이 된다"

say "1. 앵커 규약 재확인 (v10 전수)"
$PY "$REPO/AI/tools/probe_chunk_anchor.py" --dataset "$V10" --limit 0 \
    --out "$OUT/anchor.json" 2>&1 | tail -14
grep -q '"status": "ANCHOR"' "$OUT/anchor.json" && note "앵커 규약 ANCHOR 확인" \
    || note "!! 앵커 규약이 ANCHOR 가 아니다 — 변환 결과를 믿지 마라"

say "2. v10 -> 공식 UMI zarr 변환 (train 분할만)"
$PY "$REPO/AI/tools/convert_v10_to_umi.py" --dataset "$V10" \
    --out "$OUT/v10_train" --only train --holdout 14 --split-seed 42 2>&1 | tail -25
if [ -f "$OUT/v10_train.zarr.zip" ]; then
  note "변환 산출 $(stat -c%s "$OUT/v10_train.zarr.zip") B"
else
  note "!! 변환 실패 — zarr 없음. 이후 단계 미실행"; echo "E2E_0920_FINISHED_MARKER"; exit 1
fi

say "2b. 홀드아웃 분할도 따로 변환"
$PY "$REPO/AI/tools/convert_v10_to_umi.py" --dataset "$V10" \
    --out "$OUT/v10_holdout" --only holdout --holdout 14 --split-seed 42 2>&1 | tail -12
test -f "$OUT/v10_holdout.zarr.zip" && note "홀드아웃 변환 OK" || note "홀드아웃 변환 실패"

say "3. 감사 + 사전 등록 게이트"
$PY "$REPO/AI/tools/audit_umi_zarr.py" --zarr "$OUT/v10_train.zarr.zip" \
    --label v10_train --gate --rate-hz 30.0 --out "$OUT/gate_v10_train.json" 2>&1 | tail -22
$PY - "$OUT/gate_v10_train.json" <<'PYEOF' 2>&1 | tee -a "$STATUS"
import json, sys
g = json.load(open(sys.argv[1])).get("gates", {})
print(f"[STATUS] 게이트 {g.get('verdict')} · 통과 {g.get('passed')} "
      f"· 불합격 {g.get('failed')} · 미판정 {g.get('unknown')} / 전체 {g.get('total')}")
PYEOF

say "4. 학습 진입점 탐색"
TRAIN=""
for c in ~/handoff/train.py ~/handoff/umi/train.py \
         ~/hyeonseok/umi_gpu_bundle/third_party/umi/train.py \
         ~/S15P21A103_umi/third_party/umi/train.py; do
  echo "  확인 $c"
  [ -z "$TRAIN" ] && [ -f "$c" ] && TRAIN="$c"
done
if [ -z "$TRAIN" ]; then
  note "!! 학습 진입점을 못 찾았다 — 학습 **미실행**. 통과가 아니다"
  note "   변환·게이트까지는 산출됐다. $OUT 확인"
  echo "E2E_0920_FINISHED_MARKER"; exit 0
fi
note "학습 진입점 $TRAIN"

say "5. 학습  EPOCHS=$EPOCHS  GPU=$GPU"
cd "$(dirname "$TRAIN")" || exit 1
CUDA_VISIBLE_DEVICES=$GPU $PY "$(basename "$TRAIN")" \
    --config-name=train_diffusion_unet_timm_umi_workspace \
    task.dataset_path="$OUT/v10_train.zarr.zip" \
    training.num_epochs="$EPOCHS" \
    training.seed=42 \
    hydra.run.dir="$OUT/train_$STAMP" 2>&1 | tail -40
if ls "$OUT/train_$STAMP"/checkpoints/*.ckpt >/dev/null 2>&1; then
  n=$(ls "$OUT/train_$STAMP"/checkpoints/*.ckpt | wc -l)
  note "학습 산출 ckpt $n 개 · $OUT/train_$STAMP/checkpoints"
else
  note "!! ckpt 없음 — 학습 실패 또는 중단. 로그 위쪽 확인"
fi

note "종료 $(date +%Y%m%d_%H%M%S)"
say "요약"
cat "$STATUS"
echo "E2E_0920_FINISHED_MARKER"
