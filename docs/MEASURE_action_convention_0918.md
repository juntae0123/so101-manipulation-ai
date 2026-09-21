# MEASURE — 모델 출력 액션 규약 확정 (rot6d 규약 · pose 표현 · 곱 순서)

- 일자 2026-09-18 · 작성 김준태(트랙 B)
- 대상 `e2f_B_42` ckpt cfg · `~/handoff/third_party/umi` 소스
- 확신도 🟢 (스윕 + 소스 대조)

## 0. 왜 쟀나

BE 계약 `actionSpec` 의 `layout`·`rotation`·`compose` 근거가 **v10 규약**이었다.
v10 은 학습 *입력*이고 계약에 적는 건 모델 *출력*이다.

```
v10 (rot6d, 행 규약, 2026-09-17 스윕 확정)
  → convert_v10_to_umi.py 가 rotvec 으로 변환해 zarr 저장   ← 여기서 규약이 지워진다
  → 공식 UMI 가 학습 시점에 rotvec → rot6d 재확장          ← UMI 내부 규약
  → 모델 출력
```
중간에 rotvec 을 거치므로 v10 의 행 규약이 모델 출력까지 살아남는다는 보장이 없었다.

## 1. rot6d 규약 — 스윕으로 확정 🟢

`RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")` 에 정답을 아는 회전 3개.

```
[0] rotvec [0.3,-0.7,1.1]   행 0,1 이어붙임  2.220e-16  ← 최소
                            열 0,1 이어붙임  1.600e+00
                            열 0,1 교차      1.077e+00
                            행 0,1 교차      1.600e+00
[1] rotvec [0,0,0]          행 0.000e+00 · 열 0.000e+00   ← 판별력 없음
[2] rotvec [1.7,0.2,-0.9]   행 0,1 이어붙임  0.000e+00  ← 최소
                            나머지 셋        1.016 ~ 1.643
```

**UMI 내부 rot6d 도 회전행렬의 첫 두 행이다.** v10 과 일치한다.

⚠️ `[1]` 항등행렬에서는 네 후보가 전부 오차 0 이다. **회전 하나로만 쟀으면 판별한 줄 알았을 것이다.**

## 2. pose 표현 🟢

```
task.pose_repr.obs_pose_repr      = relative
task.pose_repr.action_pose_repr   = relative
```

## 3. 곱 순서 🟢

`diffusion_policy/common/pose_repr_util.py:62`
```python
elif pose_rep == 'relative':
    out = np.linalg.inv(base_pose_mat) @ pose_mat      # 학습:  A = T_cur⁻¹ @ T_next
```
양변 왼쪽에 `T_cur` → **`T_next = T_cur @ A_relative`**.

⚠️ 역방향(`backward=True`) 분기는 눈으로 확인하지 않았다. 정방향이 표현의 정의이고
역변환은 그 대수적 역이다. E1 97.0% · E3 교차 1.3% 가 왕복이 도는 기능 증거다.

## 4. layout — 소스로 확정 🟢

`umi/real_world/real_inference_util.py:179`
```python
n_robots       = action.shape[-1] // 10            # 로봇당 10차원
action_pose10d = action[..., start:start+9]        # 0..8 = 위치3 + rot6d 6
action_grip    = action[..., start+9:start+10]     # 9 = gap
pose_mat       = pose_to_mat(eef_pos[-1], eef_rot[-1])   # 기준 = 마지막 관측 = 현재 TCP
```

## 5. ⚠️ 조용히 틀릴 수 있는 경로가 셋이다

`get_real_umi_action(action, env_obs, action_pose_repr='abs')`

```
인자 안 넘김    → 'abs'        상대 궤적을 절대 pose 로 해석. 팔이 원점 근처로 간다
'rel' 로 넘김   → legacy 버그   소스 주석 원문: "legacy buggy implementation"
                               pos 를 기준 프레임으로 회전시키지 않고 단순 뺄셈,
                               rot 은 왼쪽 곱 (R_next @ inv(R_cur))
'relative'      → 정답         inv(T_cur) @ T_next
```

**`rel` 과 `relative` 가 한 글자 차이로 공존하고 셋 다 에러 없이 돈다.**
`evaluate.py` 는 `cfg.task.pose_repr.action_pose_repr` 를 명시적으로 넘겨서 살았다.
Jetson 추론 코드가 함수 이름만 알고 인자를 안 넘기면 기본값 `'abs'` 로 떨어진다.
계약 `runtimeSpec` 에 함수 이름과 **인자 값**을 같이 적는다.

## 6. 결과 — actionSpec 전 항목 실측

| 항목 | 값 | 근거 |
|---|---|---|
| `dim` | 10 | shape_meta `action.shape [10]` · 소스 `//10` |
| `horizon` | 8 | cfg 7군데 일치, 불일치 0 |
| `rateHz` | 10 (공칭) | 실제 시각은 `observation_timestamp` — **정정 2026-09-21**: "간격 불균일" 은 틀렸다. 0918 v10 74편·차분 4640개 전부 step=3 |
| `dx,dy,dz` | 상대 | `action_pose_repr = relative` |
| `rotation` | 회전행렬 첫 두 **행** | 스윕 4후보 중 오차 0 유일 |
| `compose` | `T_next = T_cur @ A_relative` | `pose_repr_util.py:62` |
| `gapUnit` | m | `gripper.csv` 실파일 0.064230841 |
| `gapRange` | [0, 0.09] | `evaluate.py` `np.clip(target[6], 0, 0.09)` |
| `execSlice` | [1, 5) | `evaluate.py:113` `absolute[1:1+action_steps]`, action_steps=4 |
| 변환 후 | 7차원 축각 절대 | `evaluate.py` `target[:3]` / `from_rotvec(target[3:6])` / `target[6]` |

## 7. 어시스턴트 오류

v10 입력 규약을 모델 출력 규약으로 옮겨 적고 BE 에 보낼 뻔했다.
사용자가 *"이거 실제값 들어가지?"* 라고 물어서 멈췄다.
결과적으로 값은 맞았지만 **맞는 값을 근거 없이 적은 것과 근거를 대고 적은 것은 다르다.**
rotvec 을 거치는 경로가 있었으므로 틀릴 수 있었다.
