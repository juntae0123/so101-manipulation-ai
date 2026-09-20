#!/usr/bin/env bash
# One-screen status of background training / probe jobs.
# 서버에서 도는 학습·프로브 잡 현황을 한 화면으로 보여준다.
set -u

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$BASE/out"
HOURS="${1:-6}"

running_names=""

printf '\n\033[1m[ 실행 중 ]\033[0m\n'
any=0
for pid in $(pgrep -f 'train_bc\.py|collect_sim\.py|probe_[a-z_]*\.py|eval_rollout\.py' 2>/dev/null); do
  comm=$(ps -o comm= -p "$pid" 2>/dev/null)
  case "${comm:-}" in *python*) ;; *) continue;; esac

  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
  [ -z "$cmd" ] && continue

  outarg=$(printf '%s' "$cmd" | grep -o -- '--out [^ ]*' | head -1 | cut -d' ' -f2)
  if [ -z "${outarg:-}" ]; then
    outarg=$(printf '%s' "$cmd" | grep -o -- '--tag [^ ]*' | head -1 | cut -d' ' -f2)
  fi
  name=$(basename "${outarg:-job-$pid}"); name="${name%.*}"
  running_names="$running_names $name"

  gpu=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
  [ -z "${gpu:-}" ] && gpu='?'

  el=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -z "${el:-}" ] && el=0

  log="$OUT/$name.out"
  prog='-'; eta='-'
  if [ -f "$log" ]; then
    p=$(grep -oE 'epoch +[0-9]+/[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+/[0-9]+')
    [ -z "${p:-}" ] && p=$(grep -oE '[a-z_]+: +[0-9]+/[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+/[0-9]+')
    if [ -n "${p:-}" ]; then
      prog="$p"
      done_n=${p%%/*}; tot_n=${p##*/}
      if [ "$done_n" -gt 0 ] 2>/dev/null; then
        eta=$(awk -v e="$el" -v d="$done_n" -v t="$tot_n" 'BEGIN{printf "%.0f분", (e/d)*(t-d)/60}')
      fi
    fi
  fi

  any=1
  printf '  GPU%-3s %-24s %-9s 경과 %-6s 남은 %-8s pid %s\n' \
    "$gpu" "$name" "$prog" "$(awk -v e="$el" 'BEGIN{printf "%.0f분", e/60}')" "$eta" "$pid"
done
[ "$any" -eq 0 ] && printf '  (없음)\n'

printf '\n\033[1m[ 최근 %s시간 내 종료 ]\033[0m\n' "$HOURS"
any=0
for log in $(find "$OUT" -maxdepth 1 -name '*.out' -mmin "-$((HOURS*60))" 2>/dev/null | sort); do
  name=$(basename "$log" .out)
  case " $running_names " in *" $name "*) continue;; esac
  any=1
  when=$(date -r "$log" '+%m-%d %H:%M' 2>/dev/null)
  printf '  \033[1m%s\033[0m  (%s)\n' "$name" "$when"
  grep -hoE 'real − blank = [-+0-9.]+|epoch +[0-9]+/[0-9]+ +train +[0-9.]+ +val +[0-9.na]+|Traceback|error:.*|rc=[0-9]+' "$log" 2>/dev/null | tail -2 | sed 's/^/      /'
done
[ "$any" -eq 0 ] && printf '  (없음)\n'

printf '\n\033[1m[ GPU ]\033[0m\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
    | awk -F', ' '($2+0) > 50 {printf "  %-3s %-10s %s  ●\n", $1, $2, $3; next} {printf "  %-3s %-10s %s\n", $1, $2, $3}'
else
  printf '  (nvidia-smi 없음)\n'
fi
printf '\n'
