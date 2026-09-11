# 실험 큐

**실행은 자동화한다. 판단은 자동화하지 않는다.**

조건은 사람이 쓴다. 러너는 큐를 비우는 일만 한다. 결과를 보고 다음 조건을 기계가
정하게 만들지 않는다 — n=100 의 95% 구간 반폭이 약 ±9%p 인데 기계가 조합을 훑고
최고를 고르면 우연히 좋은 것이 반드시 나온다. 그것은 측정이 아니라 다중비교 과적합이다.

## 상태는 디렉터리다

```
queue/pending/<name>.yaml   대기 — 사람이 여기에 쓴다
queue/done/<name>.yaml      끝났다 (게이트 통과·실패 모두)
queue/failed/<name>.yaml    실행 오류 (게이트 실패가 아니다)
queue/LEDGER.md             한 줄씩 append. 과거를 다시 쓰지 않는다
out/queue_<stamp>/<name>.log 항목별 원본 로그
```

## 항목 형식

파일명(확장자 제외)과 `name` 이 같아야 한다. `name` 은 체크포인트 `--tag` 로도 쓰여
같은 데이터로 다른 조건을 돌릴 때 덮어쓰기를 막는다.

```yaml
name: gripper_binary_v5          # = 파일명. 체크포인트 tag
prereg: docs/PREREG_xxx_0910.md  # 없거나 200바이트 미만이면 실행 거부
kind: repeat_runs
data: datasets/sim_pick_v5
runs: 3
epochs: 30
episodes: 100
seed_base: 0
eval_seed_base: 3000
action_space: joint_delta_gripper_binary   # 생략하면 configs/train/bc.yaml 값
cameras: cam_wrist               # 생략하면 데이터셋의 카메라 전부. 부분집합만 허용
device: cuda
policy_device: cpu
```

## 러너가 거부하는 것

| 거부 | 왜 |
|---|---|
| 사전등록 문서 없음·200바이트 미만 | 사전등록 없는 실험은 결과를 사후에 해석하게 된다 |
| 추적 중인 변경이 남아 있음 | 결과의 `code_sha` 가 실제 실행 코드를 안 가리킨다 |
| 모르는 `action_space` | 오타를 밤새 돌린 뒤에 알면 하룻밤이 날아간다 |
| `cam_` 으로 시작하지 않는 `cameras` 항목 | 같은 이유. 실제 존재 여부는 첫 에피소드에서 막힌다 |
| `name` ≠ 파일명 | 체크포인트 덮어쓰기 사고를 막는다 (2026-09-07 에 완주 실험을 잃었다) |

**하나라도 막히면 아무것도 시작하지 않는다.** 반쯤 돌다 멈추면 어느 조건이 어느
트리에서 돌았는지 뒤섞인다.

## 쓰는 법

```bash
# [서버]
cd ~/S15P21A103/AI
python tools/run_queue.py --dry-run     # 검증만
python tools/run_queue.py               # 순차
python tools/run_queue.py --parallel 2  # 2개 동시
```

`rc=1` 은 `repeat_runs` 의 **배포 게이트 실패**다. 실행 오류가 아니라 결과다 →
`done/` 으로 간다. `rc>1` 만 `failed/` 다.

## 병렬도

학습은 GPU 지만 평가는 **CPU + EGL 렌더**다. 경합은 CPU 에서 난다. GPU 5장이라고
5개를 띄우면 안 된다 — 서버는 공유고, 부하 165 에 렌더 스윕을 넣은 사고가 이미 있다.
**2로 먼저 재고 올린다.** 항목별로 `AI_CLAIM_NAME=queue_<name>` 이 붙어
`runtime_limits.claim` 의 중복 방지 락은 항목 단위로 걸린다.
