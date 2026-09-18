# MEASURE — 체크포인트에서 BE 계약값 실측

- 일자 2026-09-18 · 작성 김준태(트랙 B)
- 도구 `AI/tools/probe_ckpt_contract.py` · 자체검증 6/6
- 대상 `e2f_A_42` `e2f_B_42` (실 시연) · `e1_s0` (시뮬)
- 확신도 🟢 (체크포인트 실로드, cfg + state_dict 순회)

## 0. 결론

D-AI-61 ⑤ 미채움 칸이 추측 없이 채워졌다. **프로필을 둘로 나눠야 한다는 것이 실측으로 확정됐다.**

## 1. 실측값

| 필드 | 실 시연 (e2f_A/B) | 시뮬 (e1_s0) |
|---|---|---|
| `action.horizon` | **8** | **16** |
| `n_action_steps` | 8 | 8 |
| `task.obs_down_sample_steps` | **1** | **3** |
| obs `horizon` (n_obs_steps) | 2 | 2 |
| `optimizer.lr` | 0.0003 | 0.0003 |
| `policy.num_inference_steps` | 16 | 16 |
| `nParams` (model = ema_model) | **19,078,252** | **19,078,252** |
| 텐서 수 | 247 / 247 | 247 / 247 |

`action` 텐서 규격 (실 시연):
```
'action': {'shape': [10], 'horizon': 8, 'latency_steps': 0,
           'down_sample_steps': 1, 'rotation_rep': 'rotation_6d'}
```
`raw_shape` 가 없다 — obs 의 `robot0_eef_rot_axis_angle` 은 `raw_shape [3] → shape [6]` 으로
확장하지만 action 은 처음부터 10차원이다.

`horizon` 은 config 7군데에 중복 정의돼 있고 **전부 일치했다**(불일치 0).
계측기 자체검증 `[6]` 이 중복 정의를 둘 다 잡도록 되어 있어 확인 가능했다.

## 2. 프로필은 둘로 나눠야 한다 🟢

```
시뮬 프로필   horizon 16 · obs_down_sample_steps 3   (30fps 원본 → 10Hz)
실  프로필   horizon  8 · obs_down_sample_steps 1   (이미 10Hz)
```
둘 다 최종 표본은 10Hz로 수렴하지만 **예측 길이가 2배 다르다**(1.6초 vs 0.8초).

## 3. warm start 가 작동한 이유

`nParams` 가 horizon 16이든 8이든 **19,078,252 로 동일하다.**
diffusion UNet 이 시간축 1D conv 라 시퀀스 길이에 파라미터가 걸리지 않는다.
그래서 `--init-checkpoint` 로 시뮬(16) → 실(8) 가중치 이전이 shape 오류 없이 먹었고,
E2 에서 B 가 15/15 로 A 를 이겼다.

## 4. 체크포인트 크기 — B 가 A 의 2배인 이유

```
A  state_dicts = model, ema_model                 152,842,531 바이트
B  state_dicts = model, ema_model, optimizer      305,646,454 바이트

차이            152,803,923
AdamW 2모멘트   19,078,252 × 2 × 4바이트 = 152,626,016
남는 177,907    param_groups · step 등 부기
```
**파라미터 수는 A·B 동일하다.** 파일 크기로 `nParams` 를 추정했으면 2배 틀렸다.

### 배포 산출물
```
학습 ckpt 전체   306 MB  (model + ema_model + optimizer)
배포용           19,078,252 × 4바이트 = 76,313,008 바이트 ≈ 76 MB  (ema_model 만)
```
Jetson 8GB(이슈 42)에 306MB 를 통째로 올릴 이유가 없다.

## 5. 어시스턴트 오류 — 부분 증거로 "문서가 틀렸다"고 결론

e2f 체크포인트 하나만 보고 *"프로젝트 문서의 `action horizon 16 스텝 = 1.6초` 는 틀렸고
16은 `num_inference_steps` 다"* 라고 단정했다. **문서가 맞았다.**
시뮬 경로는 실제로 `action.horizon = 16` 이고 10Hz라 1.6초다. 16이 두 군데에 진짜로 있다.
E1 을 재고 나서야 드러났다.

> 같은 형태를 이 세션에 두 번 밟았다 — 0912 문서의 "대상물 규격 15~25mm"(타겟 도메인)를
> 시연 물체 폭으로 읽은 것, 그리고 이번. 둘 다 문서 한 줄을 전체 맥락 없이 인용했다.

동시에 **BE 의 `horizon: 8` 이 맞았다.** "미검증이니 빼자"는 내 제안은 철회했다.

## 6. 계측기 한계

`optimizer` state 는 중첩 구조라 최상위 텐서가 없어 `nParams 0 (텐서 0/2)` 로 찍힌다.
**모수를 같이 찍었기 때문에** "0개다"가 아니라 "평평한 텐서가 없다"로 읽혔다.

## 7. 재현

```bash
# [서버]
~/envs/handoff312/bin/python AI/tools/probe_ckpt_contract.py --selftest
~/envs/handoff312/bin/python AI/tools/probe_ckpt_contract.py --ckpt 경로 --out out.json
```

## 8. 여전히 미확정 — 추측으로 채우지 않는다

| 필드 | 상태 |
|---|---|
| `frameworkVersion` | 학습 torch 2.13.0+cu126(V100). **Jetson 추론 버전 미확인** |
| `contractVersion` | Jetson 추론 코드와 합의 필요. **보드 모델명 미확인** |
| `epochs` / `batchSize` | 이번 프로브 탐색 키에 없었다. 필요하면 키 추가 후 재실행 |
| `execSlice` index 0 의미 | D-AI-55 미해결. 트랙 A v10 경로는 0..3, 우리 실행은 1..4 |
