#!/usr/bin/env bash
# E3 -- train DAgger seeds 3/4 with the same recipe and score the fixed schedule.
# 같은 recipe 로 DAgger 학습 seed 3·4 를 추가하고 fixed 스케줄을 평가한다.
set -e
set -o pipefail
cd ~/S15P21A103/AI

export AI_THREADS=2
export MUJOCO_GL=egl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

RUNNING=$(pgrep -u "$USER" -af 'probe_gripper_schedule|repeat_runs|train_bc' | grep -v "$$" || true)
if [ -n "$RUNNING" ]; then
  echo "다른 잡이 돌고 있다. 끝난 뒤에 실행해라:" >&2
  echo "$RUNNING" >&2
  exit 1
fi

git diff --quiet -- eval policy sim configs/so101.yaml tools/probe_gripper_schedule.py

MERGED=$(find out -maxdepth 1 -type d -name 'dagger_merged_*' | sort | tail -1)
test -n "$MERGED"

EPOCHS="${EPOCHS:-30}"
TAG="dagger_extra_0910"
STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="out/dagger_extra_seeds_${STAMP}"
PREFIX="checkpoints/bc/$(basename "$MERGED")_${TAG}"
mkdir -p "$ROOT"

test ! -e "${PREFIX}_seed3.pt"
test ! -e "${PREFIX}_seed4.pt"

# recipe 대조 — seed0/1/2 와 epochs 가 다르면 5회가 아니라 다른 실험이다.
python - "$MERGED" "$EPOCHS" <<'PY'
import json
import sys
from pathlib import Path

merged, epochs = sys.argv[1], int(sys.argv[2])
found = []
for line in Path("EXP_LOG.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    if rec.get("experiment") != "train_bc":
        continue
    cond = rec.get("conditions", {})
    if str(cond.get("trained_on", "")).endswith(Path(merged).name):
        found.append((cond.get("seed"), cond.get("epochs")))

if not found:
    raise SystemExit(
        f"EXP_LOG 에 {merged} 학습 기록이 없다. 같은 데이터인지 먼저 확인해라."
    )
epochs_used = sorted({e for _, e in found})
print(f"기존 학습 기록: {found}")
if epochs_used != [epochs]:
    raise SystemExit(
        f"epochs 불일치: 기존 {epochs_used} vs 이번 {epochs}. "
        f"EPOCHS={epochs_used[0]} 로 다시 실행하거나 recipe 차이를 사전등록에 적어라."
    )
print(f"recipe fixture: PASS (epochs={epochs})")
PY

# 학습 + 롤아웃. 배포 게이트 실패(rc=1)는 여기서 정상이다.
set +e
PYTHONUNBUFFERED=1 python tools/repeat_runs.py \
  --data "$MERGED" \
  --runs 2 \
  --episodes 100 \
  --epochs "$EPOCHS" \
  --seed-base 3 \
  --eval-seed-base 3000 \
  --tag "$TAG" \
  --device cuda \
  --log \
  2>&1 | tee "$ROOT/train_and_baseline.log"
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" -gt 1 ]; then
  echo "repeat_runs 실패 rc=$RC — 게이트 실패가 아니라 실행 오류다." >&2
  exit "$RC"
fi

test -f "${PREFIX}_seed3.pt"
test -f "${PREFIX}_seed4.pt"

# repeat_runs 가 방금 측정한 seed3/4 성공률을 probe 의 fixture 로 쓴다.
EXPECTED=$(python - "${PREFIX}_seed3.pt" "${PREFIX}_seed4.pt" <<'PY'
import json
import sys
from pathlib import Path

wanted = [str(Path(p).resolve()) for p in sys.argv[1:]]
rates = {}
for line in Path("EXP_LOG.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    if rec.get("experiment") != "repeat_runs":
        continue
    for run in rec.get("result", {}).get("per_run", []):
        rates[str(Path(run["ckpt"]).resolve())] = run["rate"]

out = []
for ckpt in wanted:
    if ckpt not in rates:
        raise SystemExit(f"EXP_LOG 에 {ckpt} 의 per_run 기록이 없다")
    out.append(str(int(round(rates[ckpt] * 100))))
print(" ".join(out))
PY
)
echo "probe learned fixture (repeat_runs 재현값): $EXPECTED"
set -- $EXPECTED

python tools/probe_gripper_schedule.py \
  --policy-ckpt "${PREFIX}_seed3.pt" \
  --policy-ckpt "${PREFIX}_seed4.pt" \
  --expected "$1" \
  --expected "$2" \
  --claim-name gripper_dagger_extra_seeds \
  --episodes 100 \
  --seed-base 3000 \
  --out "$ROOT/probe" \
  --log \
  2>&1 | tee "$ROOT/probe.log"

python - "$ROOT/probe/result.json" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
root = Path(sys.argv[2])
payload = json.loads(result_path.read_text(encoding="utf-8"))

doc = [
    "# MEASURE — E3 DAgger fixed schedule 학습 seed 3·4 (2026-09-10)",
    "",
    "- 확신도: 🟢 실행·로그 확인",
    "- 사전등록: docs/PREREG_dagger_fixed_extra_seeds_0910.md",
    f"- 원본: {root}",
    "- 조건: 학습 seed 3/4, n=100/checkpoint, evaluation seeds 3000~3099",
    "",
    "| checkpoint | learned | fixed60 | fixed67 | fixed75 | fixed85 | oracle |",
    "|---|---:|---:|---:|---:|---:|---:|",
]

fixed67_pass = []
for checkpoint, values in payload["results"].items():
    def rate(name: str) -> str:
        r = values[name]
        return f"{r['success']}/{r['episodes']}"

    fixed67_pass.append(values["fixed67"]["success_rate"] > 0.20)
    doc.append(
        f"| {checkpoint} | {rate('learned')} | {rate('fixed60')} | "
        f"{rate('fixed67')} | {rate('fixed75')} | {rate('fixed85')} | "
        f"{rate('oracle')} |"
    )

doc += [
    "",
    "## 사전등록 판정",
    "",
    f"- seed3/4 fixed67 > 20%: {fixed67_pass}",
    f"- 5-run 후보 유지: {'예' if all(fixed67_pass) else '아니오'}",
    "- 기존 seed0/1/2 fixed67: 83% / 42% / 43%",
    "",
    "## 해석 제한",
    "",
    "- 학습 seed 를 5개로 늘린 것이지 평가 n 을 늘린 것이 아니다.",
    "- fixed tick 은 실물 배포 처방이 아니라 gripper phase 학습 실패 진단이다.",
    "- oracle 열은 E2 계측기 판정 결과와 함께 읽어야 한다.",
]

Path("docs/MEASURE_dagger_fixed_extra_seeds_0910.md").write_text(
    "\n".join(doc) + "\n", encoding="utf-8"
)
print("wrote docs/MEASURE_dagger_fixed_extra_seeds_0910.md")
PY

printf '\n===== E3 결과 =====\n'
cat docs/MEASURE_dagger_fixed_extra_seeds_0910.md
