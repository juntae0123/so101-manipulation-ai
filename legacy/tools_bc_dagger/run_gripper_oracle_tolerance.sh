#!/usr/bin/env bash
# E2 -- sweep the oracle close tolerance to find whether the instrument is broken.
# oracle 계측기가 고장인지 확인하기 위해 close tolerance 를 스윕한다.
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

BASE=$(
  find checkpoints/bc -maxdepth 1 \
    -name 'dagger_merged_*_dagger_segments_0909_seed0.pt' |
    sort | tail -1 |
    sed 's/_seed0\.pt$//'
)
test -n "$BASE"
echo "checkpoint base: $BASE"

STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="out/oracle_tolerance_${STAMP}"
mkdir -p "$ROOT"

python -m py_compile tools/probe_gripper_schedule.py
python tools/probe_gripper_schedule.py --help >/dev/null

{
  printf 'git_revision=%s\n' "$(git rev-parse HEAD)"
  printf 'probe_sha256=%s\n' "$(sha256sum tools/probe_gripper_schedule.py | awk '{print $1}')"
  printf 'checkpoint_base=%s\n' "$BASE"
  printf 'learned_fixture=1,0,7\n'
  printf 'tolerances_m=0.005,0.010,0.015\n'
  printf 'episodes=100/condition/checkpoint\n'
  printf 'evaluation_seeds=3000~3099\n'
  printf 'render=true\npolicy_device=cpu\njitter_m=0.05\nexecution=single_job\n'
} > "$ROOT/CONDITIONS.txt"

for SPEC in "005 0.005" "010 0.010" "015 0.015"; do
  set -- $SPEC
  NAME=$1
  TOL=$2

  python tools/probe_gripper_schedule.py \
    --policy-ckpt "${BASE}_seed0.pt" \
    --policy-ckpt "${BASE}_seed1.pt" \
    --policy-ckpt "${BASE}_seed2.pt" \
    --expected 1 \
    --expected 0 \
    --expected 7 \
    --oracle-only \
    --oracle-tolerance "$TOL" \
    --claim-name "gripper_oracle_${NAME}" \
    --episodes 100 \
    --seed-base 3000 \
    --out "$ROOT/tol_${NAME}" \
    --log \
    2>&1 | tee "$ROOT/tol_${NAME}.log"
done

python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
expected_5mm = [48, 21, 24]

for tol_name, tol_mm in (("005", 5), ("010", 10), ("015", 15)):
    payload = json.loads(
        (root / f"tol_{tol_name}" / "result.json").read_text(encoding="utf-8")
    )
    values = list(payload["results"].values())

    if tol_name == "005":
        observed = [v["oracle"]["success"] for v in values]
        if observed != expected_5mm:
            raise RuntimeError(
                f"5mm oracle reproduction failed: {observed} != {expected_5mm}"
            )

    for seed, value in enumerate(values):
        oracle = value["oracle"]
        rows.append({
            "tol_mm": tol_mm,
            "seed": seed,
            "success": oracle["success"],
            "never_closed": oracle["never_closed"],
            "close_tick": oracle["median_close_tick"],
        })

doc = [
    "# MEASURE — E2 oracle close tolerance sweep (2026-09-10)",
    "",
    "- 확신도: 🟢 실행·로그 확인",
    "- 사전등록: docs/PREREG_gripper_oracle_tolerance_0910.md",
    f"- 원본: {root}",
    "",
    "| tolerance | checkpoint | 성공률 | never closed | 중앙 close tick |",
    "|---:|---|---:|---:|---:|",
]
for row in rows:
    tick = "-" if row["close_tick"] is None else f"{row['close_tick']:.1f}"
    doc.append(
        f"| {row['tol_mm']}mm | seed{row['seed']} | "
        f"{row['success']}/100 | {row['never_closed']}/100 | {tick} |"
    )

usable = {}
for tol in (5, 10, 15):
    subset = [r for r in rows if r["tol_mm"] == tol]
    usable[tol] = (
        all(r["never_closed"] <= 5 for r in subset)
        and sum(r["success"] > 20 for r in subset) >= 2
    )

selected = next((tol for tol in (5, 10, 15) if usable[tol]), None)
doc += [
    "",
    "## 사전등록 판정",
    "",
    f"- usable: {usable}",
    f"- 선택된 최소 tolerance: {selected}mm" if selected else
    "- **어느 tolerance 도 usable 조건을 만족하지 못했다.** "
    "원인은 tolerance 가 아니다 — close 조건 정의를 다시 봐야 한다 (별도 TS).",
    "",
    "## 해석 제한",
    "",
    "- oracle 은 특권정보 진단이며 실물 배포 정책이 아니다.",
    "- tolerance 를 키워 얻은 성공률은 성능이 아니라 계측기 눈금이다.",
]

Path("docs/MEASURE_gripper_oracle_tolerance_0910.md").write_text(
    "\n".join(doc) + "\n", encoding="utf-8"
)
print("wrote docs/MEASURE_gripper_oracle_tolerance_0910.md")
PY

printf '\n===== E2 결과 =====\n'
cat docs/MEASURE_gripper_oracle_tolerance_0910.md
