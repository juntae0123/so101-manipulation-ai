#!/usr/bin/env bash
# Show what the latest queue run is doing, per condition, with elapsed time.
# 최신 큐 실행이 지금 무엇을 하고 있는지 조건별로 보여준다.
#
# 왜 있나: nohup 은 PID 만 던지고 끝이라 "언제 시작했고 몇 개가 끝났고 지금
# 무엇을 하는지" 가 안 보인다. 그걸 물을 때마다 긴 한 줄을 붙여넣게 하지 않으려고.
#
#   # [서버]
#   bash tools/st.sh          진행만
#   bash tools/st.sh -a       진행 + 합산표까지

set -u
cd ~/S15P21A103/AI || exit 1
PY=~/envs/aiot_v100/bin/python
RUNS=3          # repeat_runs 기본 반복 횟수. yaml 의 runs 와 맞춰야 한다

Q=$(ls -td out/queue_* 2>/dev/null | head -1)
if [ -z "${Q}" ]; then
  echo "큐 디렉터리가 없다 (out/queue_*). 아직 아무것도 안 돌았다."
  exit 0
fi

START=$(date -r "${Q}" '+%m-%d %H:%M')
echo "=============================================================="
echo " 큐 ${Q}"
echo " 시작 ${START}   현재 $(date '+%m-%d %H:%M')"
echo "=============================================================="

done_n=0 total_n=0
for f in "${Q}"/*.log; do
  [ -e "${f}" ] || continue
  total_n=$((total_n + 1))
  name=$(basename "${f}" .log)
  # `$ cmd` 줄은 eval/repeat.py 의 _run() 이 매 하위 프로세스마다 찍는다.
  t=$(grep -c '^\$ .*train_bc\.py' "${f}")
  r=$(grep -c '^\$ .*eval_rollout\.py' "${f}")
  if grep -q '배포 가능\|배포 불가' "${f}"; then
    state="완료"; done_n=$((done_n + 1))
  elif grep -qi 'Traceback\|Error\|error:' "${f}"; then
    state="⚠️오류"
  else
    state="진행중"
  fi
  last=$(tail -1 "${f}" | tr -d '\r' | cut -c1-38)
  printf '  %-18s %-7s 학습%s/%s 롤아웃%s/%s   %s\n' \
    "${name}" "${state}" "${t}" "${RUNS}" "${r}" "${RUNS}" "${last}"
done
echo
echo "  조건 ${done_n}/${total_n} 완료"

echo
echo "-- 지금 도는 프로세스 (경과시간) ------------------------------"
ps -o etime=,args= -u "$USER" 2>/dev/null \
  | grep -E 'train_bc\.py|eval_rollout\.py|repeat_runs' | grep -v grep \
  | sed -E 's/[^ ]*python[0-9.]* //' | cut -c1-108
if ! pgrep -u "$USER" -f 'train_bc\.py|eval_rollout\.py|repeat_runs' >/dev/null; then
  echo "  (없음 — 전부 끝났다)"
fi

if [ "${1:-}" = "-a" ]; then
  echo
  "${PY}" tools/pool_rollouts.py "${Q}"
fi
