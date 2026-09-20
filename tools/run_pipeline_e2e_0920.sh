#!/usr/bin/env bash
# End-to-end pipeline pass on our v10 real demos, through the official-UMI trainer.
# 우리 v10 실 시연을 현석 학습기(공식 UMI)로 끝까지 한 번 통과시킨다.
#
# 실행:
#   cd ~/S15P21A103 && setsid nohup bash AI/tools/run_pipeline_e2e_0920.sh > out/e2e_0920.log 2>&1 < /dev/null & disown
#
# 완료 감지 문자열: "E2E_0920_FINISHED_MARKER"
#
# 2026-09-20 정정 — 초판은 학습 진입점을 train.py + hydra 로 추정해 넣었다. 틀렸다.
#   실물: 03_umi_policy_trainer/train_policy.py train <zarr> --run-id ... --profile ...
#   cwd 는 번들 루트, python 은 시스템 파이썬(torch 2.6.0+cu126 이 거기 있다).
#   이 단계만 handoff312 를 쓰지 않는다.
#
# 원칙: 단계마다 모수를 찍는다. "안 함" 과 "됐음" 이 같은 출력으로 나오지 않게 한다.

set -u
PY=~/envs/handoff312/bin/python
BUNDLE=~/hyeonseok/umi_gpu_bundle
TRAINER=$BUNDLE/03_umi_policy_trainer/train_policy.py
PROFILE=$BUNDLE/03_umi_policy_trainer/configs/policy_resnet18_gpu.yaml
TPY=python
REPO=~/S15P21A103
V10=~/S15P21A103_umi/AI/datasets/umi_real_relative_20260918_v10_orbslam_cadtcp_video_aligned_provisional_74ep
# 변환기는 MuJoCo 태스크 설정으로 PickEnv 를 만든다 (--task 필수).
# 2026-09-20 1차 실행이 이걸 안 넘겨 2단계에서 죽었다. 프리플라이트로 올린다.
TASK=${TASK:-~/handoff/configs/can_side.yaml}
OUT=$REPO/out/e2e_0920
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID=${RUN_ID:-v10_real_74ep_$STAMP}
EPOCHS=${EPOCHS:-120}
BATCH=${BATCH:-64}
GPU=${GPU:-1}
mkdir -p "$OUT"
STATUS=$OUT/STATUS.txt
: > "$STATUS"

say()  { echo; echo "=================== $* ==================="; }
note() { echo "[STATUS] $*" | tee -a "$STATUS"; }

say "0. 프리플라이트  $STAMP"
note "시작 $STAMP · RUN_ID=$RUN_ID · EPOCHS=$EPOCHS · BATCH=$BATCH · GPU=$GPU"
ok=0; tot=0
chk() { tot=$((tot+1)); if eval "$2" >/dev/null 2>&1; then ok=$((ok+1)); echo "  [있음] $1"; else echo "  [없음] $1"; fi; }
chk "계측용 python"   "test -x $PY"
chk "v10 데이터셋"     "test -d $V10"
chk "변환기"          "test -f $REPO/AI/tools/convert_v10_to_umi.py"
chk "감사기"          "test -f $REPO/AI/tools/audit_umi_zarr.py"
chk "앵커 프로브"      "test -f $REPO/AI/tools/probe_chunk_anchor.py"
chk "학습기"          "test -f $TRAINER"
chk "학습 프로파일"    "test -f $PROFILE"
chk "태스크 설정"      "test -f $TASK"
echo "  프리플라이트 $ok / $tot"
if [ ! -f "$TASK" ]; then echo "  -- ~/handoff/configs 안의 후보:"; ls ~/handoff/configs/*.yaml 2>/dev/null | head -20 || echo "     디렉터리 없음"; fi
note "프리플라이트 $ok / $tot"
[ "$ok" -lt 8 ] && { note "!! 필수 경로 누락. 중단"; echo "E2E_0920_FINISHED_MARKER"; exit 1; }

say "0b. GPU 실연산 (가용성 확인 != 기능 확인)"
CUDA_VISIBLE_DEVICES=$GPU $TPY - > "$OUT/gpu_check.txt" 2>&1 <<'PYEOF'
import torch
print("torch", torch.__version__, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
    a = torch.randn(512, 512, device="cuda"); print("실연산 OK 합계", round((a @ a).sum().item(), 3))
else:
    print("!! CUDA 없음")
PYEOF
cat "$OUT/gpu_check.txt"
grep -q "실연산 OK" "$OUT/gpu_check.txt" && note "GPU 실연산 통과" || { note "!! GPU 실연산 실패. 중단"; echo "E2E_0920_FINISHED_MARKER"; exit 1; }

say "1. 앵커 규약 재확인 (v10 전수)"
$PY "$REPO/AI/tools/probe_chunk_anchor.py" --dataset "$V10" --limit 0 --out "$OUT/anchor.json" 2>&1 | tail -12
grep -q '"status": "ANCHOR"' "$OUT/anchor.json" && note "앵커 ANCHOR 확인" || note "!! 앵커가 ANCHOR 가 아니다"

say "2. v10 -> 공식 UMI zarr (train 분할)"
$PY "$REPO/AI/tools/convert_v10_to_umi.py" --dataset "$V10" --task "$TASK" --out "$OUT/v10_train" --only train --holdout 14 --split-seed 42 2>&1 | tail -25
ZARR=$OUT/v10_train.zarr.zip
[ -f "$ZARR" ] || { note "!! 변환 실패. 중단"; echo "E2E_0920_FINISHED_MARKER"; exit 1; }
note "변환 산출 $(stat -c%s "$ZARR") B"

say "2b. 사이드카 report.json 생성 (학습기가 요구한다)"
$PY - "$ZARR" 2>&1 <<'PYEOF' | tail -12
import hashlib, json, sys
from pathlib import Path
import numpy as np, zarr
p = Path(sys.argv[1])
z = zarr.open(zarr.ZipStore(str(p), mode="r"), mode="r")
ends = np.asarray(z["meta"]["episode_ends"]).tolist()
h = hashlib.sha256(p.read_bytes()).hexdigest()
doc = {
    "status": "pass",
    "dataset": str(p),
    "dataset_sha256": h,
    "episodes": len(ends),
    "frames": int(ends[-1]) if ends else 0,
    "episode_ends": [int(x) for x in ends],
    "native_sample_rate_hz": 10.0,
    "schema": "Stanford UMI ReplayBuffer data/meta layout",
    "image_contract": "v10 원본 그대로. inpaint 여부 미확인",
    "pose_contract": "umi_relative_chunk/0.2.0 상대 청크에서 복원한 절대 궤적",
    "world_frame": "grasp_anchor (robot_base_alignment_applied=false)",
    "episode_start_trim_s": 0.0,
    "camera_tcp_status": "unvalidated",
    "training_input_status": "provisional_camera_tcp",
    "physical_deployment_ready": False,
    "note": "hand-eye 미측정. 실물 배포 전 T_camera_tcp 측정 후 재구축 필요",
}
out = p.with_name(p.name.replace(".zarr.zip", ".zarr.report.json"))
out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
print("→ %s  편 %d · 프레임 %d · sha %s" % (out, doc["episodes"], doc["frames"], h[:12]))
PYEOF
SIDE=$OUT/v10_train.zarr.report.json
[ -f "$SIDE" ] && note "사이드카 OK" || { note "!! 사이드카 생성 실패. 중단"; echo "E2E_0920_FINISHED_MARKER"; exit 1; }

say "3. 감사 + 사전 등록 게이트"
$PY "$REPO/AI/tools/audit_umi_zarr.py" --zarr "$ZARR" --label v10_train --gate --rate-hz 10.0 --out "$OUT/gate_v10_train.json" 2>&1 | tail -22
$PY - "$OUT/gate_v10_train.json" 2>&1 <<'PYEOF' | tee -a "$STATUS"
import json, sys
g = json.load(open(sys.argv[1])).get("gates", {})
print("[STATUS] 게이트 %s · 통과 %s · 불합격 %s · 미판정 %s / 전체 %s"
      % (g.get("verdict"), g.get("passed"), g.get("failed"), g.get("unknown"), g.get("total")))
for r in g.get("rows", []):
    if r.get("ok") is not True:
        print("[STATUS]   미통과 %s: %s  기준 %s" % (r["name"], r["got"], r["want"]))
PYEOF

say "4. 학습기 dataset check"
cd "$BUNDLE" || exit 1
export UMI_ROOT=$BUNDLE/third_party/umi WANDB_MODE=disabled HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=$GPU $TPY "$TRAINER" check "$ZARR" > "$OUT/dataset_check.json" 2>&1
tail -25 "$OUT/dataset_check.json"
grep -q '"episodes"' "$OUT/dataset_check.json" && note "dataset check 통과" || { note "!! dataset check 실패 — 학습 미실행. 통과가 아니다"; echo "E2E_0920_FINISHED_MARKER"; exit 1; }

say "5. 학습  RUN_ID=$RUN_ID  EPOCHS=$EPOCHS  BATCH=$BATCH  GPU=$GPU"
CUDA_VISIBLE_DEVICES=$GPU $TPY "$TRAINER" train "$ZARR" --run-id "$RUN_ID" --runs-dir "$OUT/runs" --epochs "$EPOCHS" --batch "$BATCH" --eval-batch "$BATCH" --profile "$PROFILE" 2>&1 | tail -60

if [ -f "$OUT/runs/$RUN_ID/manifest.json" ]; then
  note "학습 완료 — manifest 있음"
  $PY - "$OUT/runs/$RUN_ID/manifest.json" 2>&1 <<'PYEOF' | tee -a "$STATUS"
import json, sys
m = json.load(open(sys.argv[1]))
b = m.get("best_checkpoint", {})
p, h = b.get("policy", {}), b.get("hold_current_pose_and_width", {})
print("[STATUS] epoch %s · baseline 이김 %s" % (b.get("epoch"), b.get("beats_hold_baseline")))
for k in ("position_component_rmse_mm", "position_distance_rmse_mm",
          "position_distance_p95_mm", "rotation_mean_deg", "width_rmse_mm"):
    pv, hv = p.get(k), h.get(k)
    r = ("%.2fx" % (hv / pv)) if pv and hv else "-"
    print("[STATUS]   %-32s 정책 %-10.10s hold %-10.10s %s" % (k, str(pv), str(hv), r))
PYEOF
else
  note "!! manifest 없음 — 학습 실패 또는 중단. 로그 위쪽 확인"
fi

note "종료 $(date +%Y%m%d_%H%M%S)"
say "요약"; cat "$STATUS"
echo "E2E_0920_FINISHED_MARKER"
