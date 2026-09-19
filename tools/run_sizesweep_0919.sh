#!/usr/bin/env bash
# 0919 학습 편수 대 시뮬 사전학습 이득 — 사전등록 docs/PREREG_sizesweep_0919.md
#
# 홀드아웃을 고정하고(seed 42, 14편 816표본) 학습 편수만 20/40/60 으로 바꿔
# A(warm start 없음) vs B(시뮬 ckpt 에서 출발)를 짝비교한다.
#
# N=60 은 f0918_A_42 / f0918_B_42 를 **재사용**한다. 새로 도는 건 4개뿐이다.
#
# 게이트 (결과 보기 전 확정):
#   G1  N=20 에서 B−A 병진 95% CI 가 0 미포함
#   G2  |B−A| 병진이 N=20 에서 N=60 보다 2.0mm 이상 크다
#
# 실행: nohup bash ~/S15P21A103/AI/tools/run_sizesweep_0919.sh > ~/handoff/outputs/sizesweep0919.log 2>&1 &
# 완료: grep -c SIZESWEEP0919_DONE ~/handoff/outputs/sizesweep0919.log

set -uo pipefail
R=$HOME/handoff
PY=$HOME/envs/handoff312/bin/python
TOOLS=$HOME/S15P21A103/AI/tools
TASK=${TASK:-configs/can_side.yaml}
CKPT_C=outputs/e2_C/checkpoints/latest.ckpt
TRAIN60=outputs/ds_f0918_42_train.zarr.zip
HOLD=outputs/ds_f0918_42_holdout.zarr.zip
SIZES=(20 40)
GPUS=(1 2 3 4)
EPOCHS=60
MIN_FREE_GB=20
cd "$R" || exit 2

echo "== 착수 전 검사 =="
for f in "$TRAIN60" "$HOLD" "$TASK" "$CKPT_C" \
         "outputs/f0918_A_42/checkpoints/latest.ckpt" \
         "outputs/f0918_B_42/checkpoints/latest.ckpt"; do
  [ -e "$f" ] || { echo "!! 없다: $f"; exit 2; }
done
grep -q -- "--override" umi_adapter/train.py || { echo "!! --override 패치 없음"; exit 2; }
grep -q -- "--init-checkpoint" umi_adapter/train.py || { echo "!! --init-checkpoint 없음"; exit 2; }
for g in "${GPUS[@]}"; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null) \
    || { echo "!! GPU $g 조회 실패"; exit 2; }
  [ -n "$u" ] && [ "$u" -lt 500 ] || { echo "!! GPU $g 가 ${u}MiB 사용 중. 남의 잡 위에 안 얹는다"; exit 2; }
done
free_gb=$(df -BG --output=avail "$R" | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge "$MIN_FREE_GB" ] || { echo "!! 디스크 ${free_gb}GB. ${MIN_FREE_GB}GB 필요"; exit 2; }
echo "   OK · 디스크 ${free_gb}GB · GPU ${GPUS[*]}"

# ── 정답을 아는 행 ────────────────────────────────────────────────────────
echo
echo "== 1. 기준 데이터 재확인 (정답 아는 행) =="
$PY - "$TRAIN60" "$HOLD" <<'PY' || { echo "!! 기준 데이터가 기대와 다르다. 조건이 달라졌다. 중단"; exit 3; }
import sys, zarr, numpy as np
exp = {sys.argv[1]: (60, 3898), sys.argv[2]: (14, 816)}
bad = 0
for p, (ee_n, rows) in exp.items():
    z = zarr.open(p, mode="r")
    e = np.asarray(z["meta"]["episode_ends"][:])
    got = (len(e), int(e[-1]))
    ok = got == (ee_n, rows)
    print(f"   {p}  편 {got[0]}/{ee_n}  행 {got[1]}/{rows}  {'OK' if ok else '!! 불일치'}")
    bad += (not ok)
sys.exit(1 if bad else 0)
PY
echo "   재현 OK"

# ── 부분집합 ─────────────────────────────────────────────────────────────
echo
echo "== 2. 학습 부분집합 생성 (중첩: 20 ⊂ 40 ⊂ 60) =="
for n in "${SIZES[@]}"; do
  dst="outputs/ds_f0918_42_train_n${n}.zarr.zip"
  if [ -e "$dst" ]; then echo "   이미 있음: $dst"; else
    $PY "$TOOLS/subset_umi_zarr.py" --src "$TRAIN60" --dst "$dst" --episodes "$n" \
      > "outputs/sz_subset_${n}.log" 2>&1 \
      || { echo "!! 부분집합 실패 (n=$n). 로그 outputs/sz_subset_${n}.log"; exit 3; }
  fi
done

echo
echo "== 3. 부분집합 검산 — **별도 프로세스에서** 다시 읽는다 =="
echo "   (zip 이 확정 안 된 채 끝나면 같은 프로세스 안에서는 안 잡힌다)"
$PY - "${SIZES[@]}" <<'PY' || { echo "!! 부분집합 검산 실패. 중단"; exit 3; }
import sys, zarr, numpy as np
bad = 0
for n in (int(x) for x in sys.argv[1:]):
    p = f"outputs/ds_f0918_42_train_n{n}.zarr.zip"
    z = zarr.open(p, mode="r")
    e = np.asarray(z["meta"]["episode_ends"][:])
    keys = sorted(z["data"].array_keys())
    rows = {k: z["data"][k].shape[0] for k in keys}
    uniq = sorted(set(rows.values()))
    ok = (len(e) == n) and (len(uniq) == 1) and (uniq[0] == int(e[-1]))
    print(f"   n={n:<3} 편 {len(e)}/{n} · 배열 {len(keys)}개 행 {uniq} · 마지막경계 {int(e[-1])}"
          f"  {'OK' if ok else '!! 경계와 행 수가 어긋난다'}")
    bad += (not ok)
sys.exit(1 if bad else 0)
PY
echo "   검산 OK"

echo
echo "== 4. 편 구성 (세션 쏠림 사후 확인용) =="
$PY - <<'PY'
import json, pathlib
p = pathlib.Path("outputs/ds_f0918_42_train.provenance.json")
if not p.exists():
    print("   provenance 없음 — 편 목록 확인 불가 (쏠림 판단 보류)")
else:
    d = json.loads(p.read_text())
    eps = d.get("episodes") or d.get("episode_ids") or d.get("kept")
    if not eps:
        print(f"   provenance 키 {sorted(d)} 에 편 목록 없음 — 쏠림 판단 보류")
    else:
        print(f"   전체 {len(eps)}편. 앞 20: {eps[:20]}")
        print(f"   앞 40 중 뒤 20: {eps[20:40]}")
PY

# ── 학습 ─────────────────────────────────────────────────────────────────
echo
echo "== 5. 학습 4개 (N=20,40 × A,B) 병렬 =="
i=0; pids=()
for n in "${SIZES[@]}"; do
  for cond in A B; do
    g=${GPUS[$i]}; out="outputs/sz_${cond}_${n}"; extra=""
    [ "$cond" = "B" ] && extra="--init-checkpoint $CKPT_C"
    if [ -e "$out" ]; then echo "   건너뜀 (이미 있음): $out"; i=$((i+1)); continue; fi
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$g" nohup $PY -m umi_adapter.train \
      --dataset "outputs/ds_f0918_42_train_n${n}.zarr.zip" --output "$out" \
      --epochs "$EPOCHS" --batch 8 --workers 4 --seed 0 --task "$TASK" $extra \
      --override task.action_horizon=8 --override task.obs_down_sample_steps=1 \
      > "outputs/sz_${cond}_${n}.log" 2>&1 &
    pids+=($!); echo "   N=$n $cond → GPU $g  pid $!"
    i=$((i+1)); sleep 3
  done
done
for p in "${pids[@]}"; do wait "$p"; done
echo "   학습 배치 완료"

echo
echo "== 6. 총 스텝 수 (에폭 고정이라 N 마다 다르다 — 귀속용) =="
for n in "${SIZES[@]}" 60; do
  for cond in A B; do
    d="outputs/sz_${cond}_${n}"; [ "$n" = "60" ] && d="outputs/f0918_${cond}_42"
    s=$(grep -oE "step[^0-9]{0,3}[0-9]+" "outputs/sz_${cond}_${n}.log" 2>/dev/null | tail -1)
    echo "   N=$n $cond  dir=$d  ${s:-스텝 미기록}"
  done
done

# ── 평가 ─────────────────────────────────────────────────────────────────
echo
echo "== 7. 같은 홀드아웃으로 평가 =="
for n in "${SIZES[@]}"; do
  for cond in A B; do
    CUDA_VISIBLE_DEVICES=1 $PY "$TOOLS/eval_holdout_error.py" \
      --checkpoint "outputs/sz_${cond}_${n}/checkpoints/latest.ckpt" \
      --dataset "$HOLD" --out "outputs/err_sz_${n}_${cond}.json" 2>&1 | tail -3
  done
done
# N=60 은 기존 평가 결과를 그대로 쓴다 (같은 홀드아웃이다)
for cond in A B; do
  src="outputs/err_f0918_42_${cond}.json"
  [ -e "$src" ] && cp "$src" "outputs/err_sz_60_${cond}.json" \
    || { echo "!! N=60 평가 결과 없다: $src"; exit 3; }
done
echo "   N=60 은 err_f0918_42_* 재사용 (홀드아웃 동일)"

echo
echo "== 8. N 별 B−A paired =="
for n in 20 40 60; do
  echo "--- N=$n ---"
  $PY "$TOOLS/eval_holdout_error.py" --compare \
     "outputs/err_sz_${n}_A.json" "outputs/err_sz_${n}_B.json"
done

echo
echo "== 9. 게이트 판정 (사전등록 그대로) =="
$PY - <<'PY'
import json, pathlib
rows = []
for n in (20, 40, 60):
    r = {}
    for c in "AB":
        p = pathlib.Path(f"outputs/err_sz_{n}_{c}.json")
        r[c] = json.loads(p.read_text()) if p.exists() else None
    rows.append((n, r))
print(f"{'N':>4} {'A trans':>9} {'B trans':>9} {'B-A':>8}   A개선%   B개선%")
d = {}
for n, r in rows:
    if not (r["A"] and r["B"]):
        print(f"{n:>4}   결과 없음"); continue
    a, b = r["A"]["chunk"]["trans_mm"], r["B"]["chunk"]["trans_mm"]
    d[n] = b - a
    print(f"{n:>4} {a:9.3f} {b:9.3f} {b-a:8.3f}   "
          f"{r['A']['improvement_pct']['trans_mm']:6.2f}  {r['B']['improvement_pct']['trans_mm']:6.2f}")
print()
if 20 in d and 60 in d:
    grew = abs(d[20]) - abs(d[60])
    print(f"G2  |B-A| N=20 {abs(d[20]):.3f} − N=60 {abs(d[60]):.3f} = {grew:+.3f} mm  "
          f"(사전등록 기준 2.0mm 이상)  → {'통과' if grew >= 2.0 else '불통과'}")
else:
    print("G2  판정 불가 — N=20 또는 N=60 결과가 없다")
print("G1  위 §8 의 N=20 병진 CI 가 0 을 포함하는지로 판정한다")
PY

echo
echo "SIZESWEEP0919_DONE"
