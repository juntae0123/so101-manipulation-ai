#!/usr/bin/env bash
# E1 -- original v5 checkpoints under fixed gripper schedule.
# 원본 v5 체크포인트를 고정 gripper 스케줄로 평가한다.
set -e
set -o pipefail
cd ~/S15P21A103/AI

export AI_THREADS=2
export MUJOCO_GL=egl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# 동시 기동 금지 (LIMITS L51). 서버는 공유고 GPU 한 장·잡 1개다.
RUNNING=$(pgrep -u "$USER" -af 'probe_gripper_schedule|repeat_runs|train_bc' | grep -v "$$" || true)
if [ -n "$RUNNING" ]; then
  echo "다른 잡이 돌고 있다. 끝난 뒤에 실행해라:" >&2
  echo "$RUNNING" >&2
  exit 1
fi

# 결과에 영향을 주는 코드·설정이 커밋되지 않았으면 실행하지 않는다.
# (code_sha 가 실제 실행 코드와 달라지는 것을 막는다)
git diff --quiet -- eval policy sim configs/so101.yaml tools/probe_gripper_schedule.py

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="out/gripper_schedule_e1_v5_${STAMP}"
LOG="out/gripper_schedule_e1_v5_${STAMP}.log"
DOC="docs/MEASURE_gripper_schedule_e1_v5_0910.md"
CONDITIONS="$OUT/CONDITIONS.txt"

mkdir -p "$OUT"

{
  printf 'git_revision=%s\n' "$(git rev-parse HEAD)"
  printf 'probe_sha256=%s\n' "$(sha256sum tools/probe_gripper_schedule.py | awk '{print $1}')"
  printf 'checkpoints=sim_pick_v5_seed0/1/2.pt\n'
  printf 'expected=14,12,2\n'
  printf 'episodes=100/condition/checkpoint\n'
  printf 'evaluation_seeds=3000~3099\n'
  printf 'render=true\n'
  printf 'policy_device=cpu\n'
  printf 'jitter_m=0.05\n'
  printf 'execution=single_job\n'
} > "$CONDITIONS"

python -m py_compile tools/probe_gripper_schedule.py
python tools/probe_gripper_schedule.py --help >/dev/null

python tools/probe_gripper_schedule.py \
  --policy-ckpt checkpoints/bc/sim_pick_v5_seed0.pt \
  --policy-ckpt checkpoints/bc/sim_pick_v5_seed1.pt \
  --policy-ckpt checkpoints/bc/sim_pick_v5_seed2.pt \
  --expected 14 \
  --expected 12 \
  --expected 2 \
  --episodes 100 \
  --seed-base 3000 \
  --out "$OUT/result" \
  --log \
  2>&1 | tee "$LOG"

python - "$OUT/result/result.json" "$CONDITIONS" "$LOG" "$DOC" <<'PY'
import json
import sys
from pathlib import Path

result_path, conditions_path, log_path, doc_path = map(Path, sys.argv[1:])
result = json.loads(result_path.read_text(encoding="utf-8"))
conditions = conditions_path.read_text(encoding="utf-8").strip()

order = ("learned", "fixed60", "fixed67", "fixed75", "fixed85", "oracle")
doc = [
    "# MEASURE — E1 원본 v5 + fixed gripper schedule (2026-09-10)",
    "",
    "- 확신도: 🟢 실행·로그 확인",
    "- 사전등록: docs/PREREG_gripper_schedule_e1_v5_0910.md",
    f"- 원본 로그: {log_path}",
    "",
    "## 조건",
    "",
    "```text",
    conditions,
    "```",
    "",
    "## 결과",
    "",
    "| checkpoint | 조건 | 성공률 | 평균 최대 상승 | 중앙 close tick | never closed |",
    "|---|---|---:|---:|---:|---:|",
]

for checkpoint, values in result["results"].items():
    for condition in order:
        row = values[condition]
        close_tick = row["median_close_tick"]
        close_text = "-" if close_tick is None else f"{close_tick:.1f}"
        never = "-" if condition == "learned" else str(row["never_closed"])
        doc.append(
            f"| {checkpoint} | {condition} | "
            f"{row['success']}/{row['episodes']} = "
            f"{100 * row['success_rate']:.1f}% | "
            f"{100 * row['mean_max_lift_m']:.2f}cm | "
            f"{close_text} | {never} |"
        )

doc += [
    "",
    "## 사전등록 판정",
    "",
    f"- fixed 통과 checkpoint 수: {result['fixed_pass_counts']}",
    f"- viable fixed: {result['viable_fixed']}",
    f"- probe 판정: {result['verdict']}",
    "",
    "## 해석",
    "",
    "- oracle 은 grasp_tol 도달 여부에 의존하며 계측기 수정(E2) 전 참고값이다.",
    "- fixed67 은 scripted 전문가의 close 위상 시작 tick 66 과 사실상 같다.",
    "- DAgger seed0 fixed75 86% 는 scripted 전체 84% 와 같은 수준이었다.",
    "- 이 결과는 시계가 좋은 제어기라는 뜻이 아니라 팔과 gripper 학습을 분리하는 진단이다.",
    "- 고정 tick 은 길이가 다른 실물 UMI 시연에 직접 적용할 수 없다.",
    "- E1 은 3-run screening 이며 정식 조건 비교에는 5회가 필요하다.",
]

doc_path.write_text("\n".join(doc) + "\n", encoding="utf-8")
print(f"wrote {doc_path}")
PY

printf '\n===== E1 결과 =====\n'
cat "$DOC"
