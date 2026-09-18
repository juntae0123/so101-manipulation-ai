# REVIEW — ARCore 어댑터 2구현 대조 + 통합안

- 일자 2026-09-12 · 작성 김준태(트랙 B)
- 대상 A: 황도경 `umi` 브랜치 (`track_a/convert/arcore.py` · `arcore_bundle.py` · `arpose_delivery.py` · `umi/gripper_gap.py`), merge `e151f953`
- 대상 B: 김준태 `umi/arcore_pilot.py` + `tools/convert_umi_pilot.py` (파일럿)
- 확신도 🟢 (양쪽 소스 전문 대조) · 🟡 통합안은 설계 판단

---

## 0. 결론

**A 가 정본이다.** B 는 폐기하고, A 가 **의도적으로 호출자에게 남긴 인자**를 채우는
얇은 호출자로 줄인다. B 가 가진 것 중 A 에 없는 것은 "값"이지 "구조"가 아니다.

단 **합치기 전에 반드시 합의해야 할 좌표계 충돌이 1건 있다 (§2).**

---

## 1. 구조 비교

| | A (황도경) | B (김준태 파일럿) |
|---|---|---|
| 입력 스키마 | `umi_raw/0.1.0` 만 | `arpose.episode/1` 직접 |
| 스키마 정규화 | `arpose_delivery.normalize_delivery()` 가 경계에서 | 없었음 → 0912 에 B 쪽에 경계 추가 (D-AI-41) |
| `T_base_world` | **호출자가 준다** (rigid 검증만) | 내부에서 정의 (중력+yaw+앵커) |
| `T_camera_pinch` | 호출자가 준다 | 내부 상수 `HANDEYE_MEASURED` |
| 좌표 변환 | `base_world @ world_camera @ camera_pinch` | 같은 식 + **`GL_TO_CV` 적용** ← §2 |
| 구간 선택 | 최장 연속 유효 런 | 폐쇄 시점 기준 앞 45 / 뒤 15 프레임 |
| 이미지 | 원본 해상도 강제 (디코더가 줄이면 거부) | 축소 허용 + `image_resize_deviation` 표기 |
| 검증 | 카메라 계약·클럭·tracking·gap 범위·프레임 조인·드롭 수·부재원 sha256 | 도달 반경 검사 정도 |
| 출처 기록 | 원본 파일별 sha256 전부 | 일부 notes |

**A 의 검증이 압도적으로 촘촘하다.** 특히 `ois/vdis/focus/ae/awb` 계약, 클럭 동일성,
`frames_dropped` 일치, 이미지 해상도 보존은 B 에 아예 없다. 이슈 30(시간동기화)의
전제를 A 는 코드로 막고 B 는 안 막는다.

## 2. ⚠️ 합의 필요 — `T_camera_pinch` 의 프레임이 서로 다르다

A 는 ARCore 포즈를 그대로 쓴다:

```python
world_camera[:3,:3], world_camera[:3,3] = quat_to_matrix(q), xyz
t = base_world @ world_camera @ camera_pinch
```

`world_camera` 는 ARCore/OpenGL 카메라 축(**+X 오른쪽 · +Y 위 · −Z 앞**)이다.
따라서 A 의 `t_camera_pinch` 도 **GL 축으로 표현돼 있어야** 한다.

그런데 HW(신현우) 회신값은 **OpenCV 축**(+X 오른쪽 · +Y 아래 · +Z 앞), 저장 JPEG 기준,
뒤집힌 실장착 기준이다 🔵. B 는 그래서 `GL_TO_CV = diag(1,-1,-1)` 을 명시적으로 적용한다.

**같은 물리 캘리브레이션인데 표현 프레임이 다르다.** OpenCV 값을 그대로 A 에 넣으면
y·z 부호가 뒤집힌 채로 궤적 전체가 틀어진다 — 그리고 **조용히** 틀어진다.
학습 손실은 잘 떨어지고 실물에서만 실패하는, 추적이 가장 어려운 종류다.

제안하는 변환 (🟡 설계 판단, 트랙 A 확인 필요):

```
M = diag(1, -1, -1, 1)
t_camera_pinch_GL = M @ T_cam_pinch_CV
```

**이건 양 트랙 접점(좌표계)이라 합의와 D- 기록이 필요하다.**
어느 쪽 축을 정본으로 할지 정하고, 한쪽에 assert 를 박아 반대쪽 값이 들어오면
실행이 멈추게 해야 한다. 부호 오류는 검증 없이는 안 드러난다.

## 3. B 가 가진 것 중 옮겨야 할 것

A 가 호출자에게 남긴 인자를 채우는 값들이다. 구조가 아니라 값이므로 **호출자로 옮긴다.**

| B 의 산출 | A 의 인자 | 근거 |
|---|---|---|
| 중력 정렬 + yaw + 앵커로 만든 `T_base_world` | `t_arcore_world_to_base` | 이슈 29 미해결 우회 🟡. 보드 생기면 재변환 |
| 실측 `T_cam→pinch` (평행이동 5,149프레임, 회전 도면+중력검정 1.2°) | `t_cam_to_pinch` | `MEASURE_handeye_from_video_0912.md` 🟢 |
| 폐쇄 시점 기준 트리밍 (앞 45 / 뒤 15) | `usable_segments` | D-AI-22. **A 의 API 로 그대로 표현된다** — 별도 코드 불필요 |
| 도달 반경 검사 (R 0.30m) | 호출자 사후 검사 | 파일럿에서 68% 가 밖으로 나갔다 🟢 |

트리밍을 `usable_segments` 로 넘기면 A 의 "최장 연속 런" 선택과 충돌하지 않는다.
A 는 주어진 구간 안에서만 고르기 때문이다.

## 4. 양쪽 다 처리하지 않은 것 — 180° 회전

0911 실측 🟢 으로 **폰이 거꾸로 장착돼 프레임이 180도 돌아 있다.**
`tools/probe_vlm_m0.py` 는 읽을 때 `img.rotate(180)` 을 한다.
**A 도 B 도 변환 단계에서는 회전하지 않는다.**

따라서 `datasets/umi_real_20260911_v2` 의 이미지는 뒤집힌 상태다.
BC 에는 그 자체로 문제가 아니다 — **실물 로봇 손목 카메라가 같은 방향으로 장착되면** 된다.
문제는 그게 아직 확인되지 않았다는 것이다.

**HW 확인 필요:** 로봇팔 손목 카메라의 장착 방향이 시연 폰과 같은가.
다르면 수집분 전체를 회전시켜 재변환해야 하고, 그 판단은 지금 해야 싸다.

## 5. 통합 절차

1. **먼저** §2 좌표계 합의 + D- 기록
2. 도경 `umi` 브랜치를 `ai` 에 머지
3. `umi/arcore_pilot.py` 삭제, `tools/convert_umi_pilot.py` 를 A 호출자로 재작성
   - `T_base_world` 구성(§3)만 남긴다
   - `t_cam_to_pinch` 는 합의된 축으로 변환해 넘긴다
   - 트리밍은 `usable_segments` 로 표현
4. `umi/raw.py` 의 `BUNDLE_SCHEMA_SUPPORTED` 는 `("umi_raw/0.1.0",)` 유지 (D-AI-41, 완료)
5. B 의 임시 경계 어댑터 블록 제거
6. 파일럿 변환 결과를 A 경로로 재생성하고 **도달 반경 검사를 다시 통과**시킨다

## 6. 이미 정리된 것

- `AI/tools/make_gripper_csv.py` 폐기 — 도경 `umi/gripper_gap.py` 가 정본
  (`MEASURE_gripper_crosscheck_0912.md`). 그쪽엔 `SCALE_CORRECTION` 이 없다.
  내 보정계수는 마스크 차이를 상수로 덮으려던 것이고 일반화되지 않았다
- `BUNDLE_SCHEMA_SUPPORTED` 되돌림 완료 (D-AI-41)
- 도경 `arcore_bundle.decode_bundle` 은 이미 `umi_raw/0.1.0` 만 받는다 — D-AI-41 과 일치한다
