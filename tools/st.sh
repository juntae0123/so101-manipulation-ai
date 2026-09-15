#!/usr/bin/env bash
# Show what the latest queue run is doing, per condition, with elapsed time.
# 최신 큐 실행이 지금 무엇을 하고 있는지 조건별로 보여준다.
#
# 왜 있나: nohup 은 PID 만 던지고 끝이라 "언제 시작했고 몇 개가 끝났고 지금
# 무엇을 하는지" 가 안 보인다. 그걸 물을 때마다 긴 한 줄을 붙여넣게 하지 않으려고.
#
# ⚠️ 정정 2026-09-15 — 완료 판정에 `배포 불가` 를 썼다가 전 조건이 즉시 "완료" 로
#    나왔다. `eval/rollout.py:447` 이 특권 baseline 마다 "(특권정보 사용 — 실물
#    배포 불가)" 를 찍기 때문이다. 부분 문자열이 다른 맥락에서 나오는 것을 종료
#    신호로 쓴 것이다. `eval/repeat.py` 만 찍는 "배포 게이트 판정" 으로 바꿨다.
#
#   # [서버]
#   bash tools/st.sh          진행만
#   bash tools/st.sh -a       진행 + 합산표까지

set -u
cd ~/S15P21A103/AI || exit 1
PY=~/envs/aiot_v100/bin/python
RUNS=3          # repeat_runs 기본 반복 횟수. yaml 의 runs 와 맞춰야 한다

# repeat.py 가 전체 반복을 마칠 때만 찍는 줄. rollout 단건 출력과 겹치지 않는다.
DONE_MARK='배포 게이트 판정'

Q=$(ls -td out/queue_* 2>/dev/null | head -1)
if [ -z "${Q}" ]; then
  echo "큐 디렉터리가 없다 (out/queue_*). 아직 아무것도 안 돌았다."
  exit 0
fi

T0=$(stat -c %Y "${Q}")
NOW=$(date +%s)
EL=$((NOW - T0))
printf '==============================================================\n'
printf ' 큐 %s\n' "${Q}"
printf ' 시작 %s   현재 %s   경과 %dh%02dm\n' \
  "$(date -d "@${T0}" '+%m-%d %H:%M')" "$(date '+%m-%d %H:%M')" \
  $((EL / 3600)) $(((EL % 3600) / 60))
printf '==============================================================\n'

done_n=0; total_n=0; steps_done=0; steps_all=0
for f in "${Q}"/*.log; do
  [ -e "${f}" ] || continue
  total_n=$((total_n + 1))
  name=$(basename "${f}" .log)
  # `$ cmd` 줄은 eval/repeat.py 의 _run() 이 하위 프로세스마다 하나씩 찍는다.
  t=$(grep -c '^\$ .*train_bc\.py' "${f}")
  r=$(grep -c '^\$ .*eval_rollout\.py' "${f}")
  steps_done=$((steps_done + t + r))
  steps_all=$((steps_all + 2 * RUNS))
  if grep -q "${DONE_MARK}" "${f}"; then
    state="완료"; done_n=$((done_n + 1))
  elif grep -qi 'Traceback\|SystemExit\|Error:' "${f}"; then
    state="⚠️오류"
  elif [ "${r}" -ge "${t}" ] && [ "${r}" -gt 0 ]; then
    state="롤아웃"
  else
    state="학습"
  fi
  last=$(tail -1 "${f}" | tr -d '\r' | cut -c1-40)
  printf '  %-16s %-7s 학습%s/%s 롤아웃%s/%s  %s\n' \
    "${name}" "${state}" "${t}" "${RUNS}" "${r}" "${RUNS}" "${last}"
done

echo
printf '  조건 %d/%d 완료 · 단계 %d/%d' "${done_n}" "${total_n}" "${steps_done}" "${steps_all}"
# 대략적인 잔여 시간. 단계 길이가 균일하다고 가정한 추정치이지 측정값이 아니다.
if [ "${steps_done}" -gt 0 ] && [ "${steps_done}" -lt "${steps_all}" ]; then
  eta=$((EL * (steps_all - steps_done) / steps_done))
  printf ' · 남은 시간 대략 %dh%02dm 🟡추정' $((eta / 3600)) $(((eta % 3600) / 60))
fi
echo

echo
echo "-- 지금 도는 프로세스 (경과시간) ------------------------------"
ps -o etime=,args= -u "$USER" 2>/dev/null \
  | grep -E 'train_bc\.py|eval_rollout\.py|repeat_runs' | grep -v grep \
  | sed -E 's/[^ ]*python[0-9.]* //; s/--data [^ ]*//; s|/home/[^ ]*/checkpoints/||' \
  | cut -c1-104
if ! pgrep -u "$USER" -f 'train_bc\.py|eval_rollout\.py|repeat_runs' >/dev/null; then
  echo "  (없음 — 전부 끝났다)"
fi

if [ "${1:-}" = "-a" ]; then
  echo
  "${PY}" tools/pool_rollouts.py "${Q}"
fi
