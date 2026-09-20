#!/usr/bin/env bash
set -e
set -o pipefail
cd ~/S15P21A103/AI

export AI_THREADS=2
export MUJOCO_GL=egl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

cat > docs/PREREG_gripper_schedule_0909.md <<'MD'
# PREREG — DAgger 정책 gripper schedule probe (2026-09-09)

## 근거

DAgger 정책은 물체 최근접 1.5~1.6mm, 턱 접촉 83~94%까지 개선됐지만
45~54/100편에서 닫지 않았다.

## 조건

- checkpoints: DAgger segment seed0/1/2
- n=100/조건/checkpoint, 평가 seed 3000~3099
- render=True, policy-device=cpu, jitter ±50mm, max 200 ticks
- arm action은 BC 출력을 그대로 사용
- learned: gripper도 BC 출력
- fixed60/67/75/85: 해당 tick 전 open, 이후 close로 강제
- oracle: 파지점과 목표 파지 위치의 3D 거리가 5mm 이하가 되면 close
- oracle은 진단용 특권정보이며 배포 불가

## 계측기 게이트

learned 조건이 직전 결과 seed0/1/2 = 1/0/7을 정확히 재현해야 한다.

## 판정

- 같은 fixed tick이 3개 checkpoint 중 2개 이상에서
  성공률 >20%이고 learned 대비 >=20%p 개선하면 clock schedule 후보
- oracle이 2개 이상에서 같은 기준을 넘고 fixed는 못 넘으면
  시각 기반 close trigger가 필요
- 둘 다 못 넘으면 gripper 단독 문제가 아니며 lift/phase arm action을 검사
MD

python -m py_compile tools/probe_gripper_schedule.py

DAGGER_DIR=$(
  find checkpoints/bc -maxdepth 1 \
    -name 'dagger_merged_*_dagger_segments_0909_seed0.pt' |
    sort | tail -1 |
    sed 's/_seed0\.pt$//'
)
test -n "$DAGGER_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="out/gripper_schedule_${STAMP}"
LOG="out/gripper_schedule_${STAMP}.log"

python tools/probe_gripper_schedule.py \
  --policy-ckpt "${DAGGER_DIR}_seed0.pt" \
  --policy-ckpt "${DAGGER_DIR}_seed1.pt" \
  --policy-ckpt "${DAGGER_DIR}_seed2.pt" \
  --expected 1 \
  --expected 0 \
  --expected 7 \
  --episodes 100 \
  --seed-base 3000 \
  --out "$OUT" \
  --log \
  2>&1 | tee "$LOG"

python - "$OUT/result.json" "$LOG" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
conditions = ("learned", "fixed60", "fixed67", "fixed75", "fixed85", "oracle")

doc = [
    "# MEASURE — DAgger gripper schedule probe (2026-09-10)",
    "",
    "- 확신도: 🟢 실행·로그 확인",
    f"- 원본 로그: `{sys.argv[2]}`",
    "- 조건: n=100/조건/checkpoint, seed 3000~3099, render=True, policy CPU",
    "",
    "| checkpoint | 조건 | 성공률 | 평균 최대 상승 | 중앙 close tick |",
    "|---|---|---:|---:|---:|",
]

for checkpoint, values in result["results"].items():
    for condition in conditions:
        r = values[condition]
        tick = "-" if r["median_close_tick"] is None else f"{r['median_close_tick']:.0f}"
        doc.append(
            f"| {checkpoint} | {condition} | "
            f"{r['success']}/{r['episodes']} = "
            f"{100*r['success_rate']:.1f}% | "
            f"{100*r['mean_max_lift_m']:.2f}cm | {tick} |"
        )

doc += [
    "",
    "## 사전등록 판정",
    "",
    f"- fixed 통과 checkpoint 수: {result['fixed_pass_counts']}",
    f"- oracle 통과 checkpoint 수: {result['oracle_pass_count']}",
    f"- viable fixed: {result['viable_fixed']}",
    f"- 판정: **{result['verdict']}**",
]

Path("docs/MEASURE_gripper_schedule_0910.md").write_text(
    "\n".join(doc), encoding="utf-8"
)
PY

python tracking/findings.py

printf '\n===== 결과 =====\n'
cat docs/MEASURE_gripper_schedule_0910.md
