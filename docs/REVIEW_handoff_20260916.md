# REVIEW — 전달 패키지 `handoff_20260916` 분석

- 작성 김준태(트랙 B) · 2026-09-16 · 이슈 S15P21A103-34, -63, -27
- 대상 `C:\Users\SSAFY\Downloads\handoff_20260916\handoff` (512개 항목)
- 상태 🔵 **문서·코드 정독.** 실행 검증은 n=100 롤아웃 대기 중

---

## 0. 한 줄

**공식 UMI → Diffusion Policy → MuJoCo 시연까지 완결된 파이프라인이 통째로 왔다.**
다만 검증은 **n=1** 이고, 회전 표현이 우리 결정과 다르다.

## 1. 구성

```
assets/robot/       SO-101 원본 + ver1 그리퍼 + 폰 홀더 (URDF·메시·Blender·좌표 명세)
assets/ycb/         005_tomato_soup_can (원본 Google 16k 스캔, CC BY 4.0)
assets/workshop/    작업대·벽·선반 텍스처
simulation/         씬 생성·센서 기록·전문가·평가 (MuJoCo)
umi_adapter/        센서→UMI 데이터셋 변환·학습·SLAM 연결
checkpoints/        encoder_latest.ckpt 152MB · Diffusion Policy · 30 epochs
third_party/        ORB_SLAM3 · 공식 UMI (고정 버전, 라이선스 포함)
examples/           seed 5000 시연 영상 + evaluation.json
```

## 2. 검증 범위 — ⚠️ n=1 이다

`VALIDATION.md` 가 스스로 적어둔 것:

```
seed 5000: 성공, 최대 리프트 0.1212867 m, 유지 0.51 s
한 번의 시연 결과이며 일반적인 성공률 100%를 의미하지 않습니다.
현재 전달 작업에서 학습이나 정책 시연을 다시 실행하지 않았습니다.
```

`evaluation.json` 의 `success_rate: 1.0` 은 **1/1** 이다.
이 프로젝트 기준으로는 판정 불가다 — n=20 도 부족해서 n=100 을 쓴다.

문서가 한계를 먼저 적어두었다는 점은 신뢰할 만하다. 남은 것은 우리가 재는 것뿐이다.

추가로 명시된 한계:
- **카메라 내부·외부 파라미터가 실제 S22 측정값이 아닌 임시 설정**
- 근거리 클리핑·카메라 배치 문제가 남아 있음
- 폭 센서는 이상적인 시뮬 관절 읽기 (실제 센서 아님)
- 물리 로봇 통신 드라이버 미포함

## 3. ⚠️ 회전 표현이 셋으로 갈라져 있다 — 제일 중요

`evaluation.json` 의 정책 입력 키:

```
camera0_rgb
robot0_gripper_width
robot0_eef_pos
robot0_eef_rot_axis_angle
robot0_eef_rot_axis_angle_wrt_start
```

```
handoff / 공식 UMI   axis-angle (3) + wrt_start     <- 체크포인트가 이걸로 학습됨
트랙 A v6            6D 회전 (회전행렬 첫 두 행)      <- 0915 합의
트랙 B 계약 0.3.0     관절각                          <- IK 이후
```

**세 경로가 서로 다른 회전 표현을 쓴다.** 셋 다 SO(3) 를 표현하지만 학습 타깃으로서
성질이 다르다 — 6D 는 연속이고, axis-angle 은 회전각 π 근처에서 불연속이다.

⚠️ **공식 UMI 체크포인트를 쓸 거라면 그 입력 포맷을 따라야 한다.**
0915 에 확정한 6D 회전(첫 두 행 + Gram-Schmidt)을 그대로 두면 이 체크포인트를
재사용할 수 없고, 데이터셋도 따로 만들어야 한다. 양 트랙 재확인 사항이다.

## 4. 정정 — 도달영역 "모순" 은 내 오독이었다 🔴

처음에 `workspace_x: [0.385, 0.415]` 를 보고 내 실측 `x [0.10, 0.25]` 와
모순이라고 판단했다. **틀렸다.**

`evaluation.json` 이 기록한 실제 task 가 `ycb_can_slam` 이고, 그 조건에서
**x 0.385~0.415 에 실제로 도달했다.**

설명: 내 `x [0.10, 0.25]` 는 **수직 하강 파지 + wrist_roll 0 고정** 조건의 수치다
(MEASURE_mujoco_scene_0827). 스캔 범위 자체는 x 0.05~0.40 이었다.
**측면 접근은 팔을 수평으로 뻗으므로 더 멀리 간다.** 같은 로봇도 파지 자세가
다르면 도달 영역이 다르다. 모순이 아니라 다른 조건이다.

→ **공짜로 생긴 교차검증**: ver1 측면 파지 도달영역 스캔 (d) 조건
(`jaw 수평 강제 + 접근축 수평 soft`)이 x 0.385~0.415 를 도달 가능으로 뱉으면
스캔이 맞는 것이고, 아니면 내 IK 제약이 틀린 것이다.

## 5. 씬·태스크 상수

```
물체      YCB 005_tomato_soup_can · 349g · 지름 66mm · 높이 101mm
          (config: radius 0.0339 · half_height 0.0509275 · mass 0.349)
그리퍼    open_width 0.09 · close_width 0.045   -> 지름 66mm 를 45mm 로 압착
성공 판정  lift_height 0.13 m · success_max_width 0.085
로봇 받침  base_height 0.0 (씬 기본값은 0.15)
배치      workspace_x [0.385, 0.415] · y [-0.035, 0.035] · yaw 전 범위
실행      action_steps 4 · episode_time_limit 12s (실제 5.8s 에 성공)
```

⚠️ `run_demo.ps1` 은 `--task` 를 넘기지 않는다. 그대로 돌리면 `env.py` 기본값
(`workspace_x [0.23, 0.29]`)으로 돌아 **evaluation.json 과 다른 조건**이 된다.
재현하려면 `--task configs/can_slam.yaml` 을 명시해야 한다.

## 6. action horizon — 트랙 B 결과와 대조 🟢

```
공식 UMI (handoff)   action_steps 4 @ 10Hz  =  0.4 초
트랙 B 최적 (실측)    K=8 @ 30Hz            =  0.267 초   <- 단봉의 꼭대기
트랙 A v6 (잠정)      K=8 @ 10Hz            =  0.8 초
```

내 K 스윕(MEASURE_chunk_k_sweep_0915, n=300/조건)에서 0.267초가 최적이었고
0.533초는 구간 분리로 낮았다. **공식 UMI 의 0.4초는 내 최적과 같은 자릿수이고,
v6 의 0.8초는 2배다.**

다른 데이터·다른 물체·다른 타깃 표현이라 그대로 적용되지 않는다. 다만 v6 의
K=8 @10Hz 가 잠정값이라면, **공식 UMI 도 트랙 B 도 더 짧은 horizon 을 쓰고 있다**는
점은 재검토 근거가 된다.

## 7. 물체가 셋으로 갈라져 있다

```
트랙 B 시뮬    2cm 20g 큐브, 상단 파지        <- 내 모든 수치의 조건
트랙 A         구운감자 케이스, 측면 파지      <- 실측 폭 없음 (39mm 잠정)
handoff        YCB 토마토 캔, 측면 파지        <- 실측 있음 (66mm, 349g)
```

실물 로봇 모터 고장으로 시뮬이 메인 경로가 된 지금, **어느 물체가 출품 대상인지**가
정해져야 씬·도달영역·파지 자세·시연 프로토콜이 전부 따라온다. PM 결정 사안이다.

참고 — YCB 캔은 **실측 물리값이 공개되어 있고**(Calli et al. ICAR 2015 Table I)
메시도 원본 스캔이라, 구운감자 케이스보다 재현성 면에서 유리하다.
다만 지름 66mm 는 `close_width 45mm` 로 압착하는 값이고, SO-101 실물 그리퍼
개구(실사용 5.3~79.4mm, MEASURE_gripper_closure_0912)로는 잡을 수 있다.

## 8. 다음

| | |
|---|---|
| 즉시 | n=100 롤아웃 (`--seed 6000 --episodes 100 --task configs/can_slam.yaml`) |
| 그다음 | ver1 측면 파지 도달영역 (d) 스캔 → x 0.385~0.415 교차검증 |
| 양 트랙 | **회전 표현 3종 갈라짐 해소.** D-AI 재확인 |
| PM | 출품 대상 물체 확정 |

⚠️ n=100 은 **seed 5000 을 피해 6000 부터** 쓴다. 5000 은 성공이 확인된 시드라
포함하면 편향된다.
