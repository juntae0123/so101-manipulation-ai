# DEVLOG 2026-09-18 — 실 시연 90편 선별 · 액션 규약 확정 · 실데이터 첫 정책

작성 김준태(트랙 B) · 확신도는 항목마다 표기 · **전부 시뮬 또는 오프라인 지표.**

⚠️ **정정**: 실물 로봇팔은 2026-09-18 모터 교체로 **복구됐다**(D-AI-58). 실물 평가가
아직 0건인 이유는 고장이 아니라 **1a·1b 안전 절차(설치물 간섭·TCP/베이스 정렬·
속도/가속도·비상정지) 승인 대기**다. 초판에 "모터 고장으로 평가 불가"로 적었던 것을 정정한다.

> 이 사이클은 달력 이틀에 걸쳐 있다. **전반부**(E2 folds · 실물 기하 · K 스윕)는
> `MEASURE_e2_*`, `MEASURE_real_geometry_0918.md`, `TS_assistant_errors_0918.md` 가 담는다.
> 이 문서는 **후반부** — 수집 선별부터 실데이터 첫 정책까지다.

---

## 0. 하루의 한 줄

**계측기 5개를 새로 만들어 실 시연 92편을 학습 가능한 79편으로 좁혔고,
모델 출력 액션 규약 전 항목을 소스·스윕으로 확정해 BE 계약을 닫았다.
그 데이터로 실데이터 첫 정책을 학습해 identity 대비 trans 69.2% 개선을 받았다 (n=1).**

막힌 것: 5-fold 로 n 을 올리는 실행이 아직 안 끝났다. **단일 fold 수치는 인용 금지.**

---

## 1. 수집 품질 선별기 — `AI/tools/check_collection_quality.py` 🟢

### 왜 만들었나

도경(트랙 A)이 SLAM 을 돌리기 전에 **우리가 먼저 거른다**는 결정.
SLAM 이 비싼 단계이므로 명백한 불량을 앞에서 떨어뜨리는 게 싸다.

### 게이트와 관측의 분리

판정에 쓰는 것과 그냥 보는 것을 섞지 않았다.

```
게이트   SYNC 불일치 > 1us · IMU 커버리지 · 마커 검출 0건 · 길이 < 3.0s
관측만   marker_rate · longest_miss · drops · exposure_cv
```

관측 항목을 게이트에 넣지 않은 이유: 우리 마커 검출기의 검출률이
**도경 파이프라인의 검출률과 같다는 보장이 없다.** 우리 숫자로 임계를 박으면
남의 파이프라인을 우리 검출기로 판정하게 된다.

### 결과 (stride 1 전수)

```
합격        90 / 92
동기화      92/92 불일치 0us
IMU         92/92 포함
진짜 드롭   0건
마커(우리)  중앙 0.726 [0.548, 0.856] · 최장 미검출 중앙 8 최대 28프레임
세션        둘로 분리 (18편 / 72편) — 분할할 때 층이 필요하다
```

### 도중에 잡은 계측기 결함 — `drops` 가 92/92 전부 1

전부 1 이면 그건 신호가 아니라 상수다. 파보니 **카메라 첫 프레임 간격이 정확히
공칭의 2배**였고(`first_gap_ratio` min=max=2.00, 92편 전수 동일), 워밍업 아티팩트다.

```python
def drop_rate(ts_ns: list[int], nominal_ns: int) -> tuple[int, int]:
    """Gaps longer than 1.5x nominal, EXCLUDING the first interval.
    0918 수집분 92편 전부에서 index 0 간격이 정확히 공칭의 2배로 나왔다."""
    if len(ts_ns) < 3 or nominal_ns <= 0:
        return (0, 0)
    gaps = [ts_ns[i + 1] - ts_ns[i] for i in range(len(ts_ns) - 1)][1:]
    return (sum(1 for g in gaps if g > nominal_ns * 1.5), len(gaps))
```

첫 간격은 버리는 대신 `first_gap_ratio` 로 **따로 보고**한다. 버린 걸 안 보이게
하면 다음 사람이 같은 걸 또 판다. 자체검증 `[5b]` `[5c]` 추가.

> **모수를 같이 찍은 덕에 드러났다.** `1건` 만 찍었으면 "드롭이 좀 있네"로 넘어갔다.
> `1 / 전체` 가 92편 전부 같은 값이라 상수라는 게 보였다.

---

## 2. 그리퍼 gap 판정기 — `AI/tools/check_gap_validity.py` 🟢

### 6일 묵은 미결이 자 하나로 닫혔다

0912 에 남긴 (A)/(B) 양자택일 — *"39mm 는 닫힌 건가 열린 건가"* — 를
**물체 실폭 41mm**(김현석 실측) 하나가 (A) 로 확정했다.

```
도경 0911 정본        37.26mm
HW  v4.zarr           39.13mm
HW  s22_umi_v3 81편   39.13mm
물체 실폭             41.00mm
```

세 독립 측정이 2mm 안에서 일치하고 물체 폭 아래다 → **그리퍼는 닫히고 있었다.**

**0912 의 결론 3개를 철회한다:**
- ~~"그리퍼가 안 닫힌다"~~
- ~~"39mm 는 여전히 열린 상태"~~
- ~~"그리퍼 채널은 의미가 없다"~~

계측기를 더 만들 문제가 아니라 **숫자 하나를 물어볼 문제였다.**

### 게이트

```
G1  골 깊이 비 <= 0.35        이봉성 (열림/닫힘이 갈리는가)
G2  편별 최소 gap 의 중앙값
G3  |G2 - object_width| <= 10mm
```

`--object-width-mm` 없이 돌리면 **"판정 불가"** 로 멈춘다. 절대 통과시키지 않는다.

### G1 초판을 자체검증이 기각했다

초판은 Otsu 분리도 >= 2.0 이었다. **단봉 정규분포를 넣었더니 2.63 으로 통과했다.**
Otsu 는 무엇을 넣어도 둘로 쪼개므로 이봉성 판정에 쓸 수 없다.
→ 골 깊이 비로 교체. **실데이터를 보기 전에 잡혔다.**

### 정답 아는 코퍼스 회귀

0911 도경 정본 71편에 돌려 **검출 5759/6087 = 0.9461 · 편별 최소 gap 중앙 37.261mm**.
6일 전 0912 측정과 소수점까지 일치. 계측기가 옛 결과를 재현한다.

---

## 3. 체크포인트 계약 프로브 — `AI/tools/probe_ckpt_contract.py` 🟢

cfg 경로를 가정하지 않고 **재귀 탐색**한다. 경로를 하드코딩하면 구조가 바뀔 때
"없음"이 조용히 나온다.

```python
WANTED = ("horizon", "n_action_steps", "n_obs_steps", "obs_down_sample_steps",
          "lr", "learning_rate", "num_inference_steps", "shape_meta",
          "pose_repr", "obs_pose_repr", "action_pose_repr",
          "rotation_rep", "rotation_transformer")
```

자체검증 `[6]` 은 **같은 키가 두 군데 정의된 경우 둘 다 잡는지**를 본다.
하나만 잡고 멈추면 불일치를 못 본다.

### 실측

```
nParams          19,078,252 (실·시뮬 동일)
lr               3e-4 · num_inference_steps 16 · n_obs_steps 2
horizon          실 8 / 시뮬 16
obs_down_sample  실 1 / 시뮬 3
ckpt 바이트       A 152,842,531  B 305,646,454   차이 = AdamW 2모멘트
배포용            ema_model 만 76MB
```

**`horizon` 과 `obs_down_sample_steps` 는 독립 축이다.** 후자는 입력 레이트가 정하고
전자는 설계 선택이다. 한 덩어리 "프로파일"로 묶어 생각했던 걸 정정한다.

warm start 가 되는 이유도 여기서 나왔다 — **nParams 가 horizon 과 무관하다.**
diffusion UNet 이 시간축 1D conv 라 horizon 8 이든 16 이든 19,078,252 로 같다.

---

## 4. 모델 출력 액션 규약 확정 🟢 → `MEASURE_action_convention_0918.md`

BE 계약에 적을 값의 근거가 **v10 규약**이었는데 v10 은 학습 *입력*이다.
중간에 `rot6d → rotvec → rot6d` 를 거치므로 규약이 살아남는다는 보장이 없었다.

스윕으로 확인: **UMI 내부 rot6d 도 회전행렬 첫 두 행.** 오차 2.220e-16, 나머지 세
후보는 1.02~1.64. v10 과 일치했다 — 하지만 **맞는 값을 근거 없이 적는 것과
근거를 대고 적는 것은 다르다.**

항등회전 표본에서는 네 후보가 전부 오차 0 이었다. **판별력 없는 표본을 섞어 넣은 게
그걸 드러냈다.** 회전 하나로만 쟀으면 판별한 줄 알았을 것이다.

### 조용히 틀리는 경로 셋 — 계약에 박았다

```
인자 안 넘김   → 'abs'       상대를 절대로 해석. 팔이 원점으로 간다
'rel'          → legacy 버그  소스 주석 원문 "legacy buggy implementation"
'relative'     → 정답
```

**셋 다 에러 없이 돈다.** `evaluate.py` 는 cfg 값을 명시적으로 넘겨서 살았다.
Jetson 추론 코드가 함수 이름만 알고 인자를 안 넘기면 기본값 `'abs'` 로 떨어진다.

---

## 5. BE 계약 — `actionSpec` 전 항목에 출처가 붙었다 🟢

```
dim 10 · horizon 8 · rateHz 10(공칭) · dx,dy,dz 상대
rotation = 회전행렬 첫 두 행 · compose T_next = T_cur @ A_relative
gapUnit m · gapRange [0, 0.09] · execSlice [1,5) · 변환 후 7차원 축각 절대
```

미정으로 남은 것: `epochs` · `batchSize` 실측, Jetson 확정 후 `frameworkVersion`
· `contractVersion`, `profileKey`/`algorithm` enum 문자열 확인.

---

## 6. 트랙 분담과 재현 대조 규약 — D-AI-62

`v10 → 공식 UMI Zarr` 변환을 트랙 B 가 실행한다. 상세는
`DECISIONS_AI_0918_pipeline_split.md`.

**초안은 2단계 전달이었고 사용자가 뒤집었다** — *"이건 진짜 왔다갔다 하는거잖아"*.
맞는 지적이다. 비싼 단계는 SLAM 인데 나는 싼 단계(v10 변환)에 게이트를 걸었다.
왕복 비용이 아낀 시간보다 컸다. 일회성 전달 + 비동기 검증으로 바꿨다.

검사기 `AI/tools/check_zarr_parity.py` 작성. 세 상태로 판정한다 —
`BIT_IDENTICAL` / `NUMERICALLY_DIFFERENT` / `SCHEMA_MISMATCH`.
`camera0_rgb` 는 게이트에서 제외(리사이즈·디코더 차이), numpy/scipy/cv2/BLAS/스레드
환경을 같이 기록해 **불일치를 환경에 귀속할 수 있게** 했다. 도경 수용.

---

## 7. SLAM 배치 요약 — 학습 가능 편수는 90 이 아니다 🟢

`AI/tools/summarize_slam_batch.py` 로 HW 배치를 우리 합격 목록과 교차했다.

```
우리 선별 합격        90 / 92
그중 SLAM 실패        9편   tracked_ratio 0.034~0.19 · keyframes 전부 0
                            궤적 길이 최대 7.24m = 발산
→ 학습 가능            81편
비파지 2편 제외        79편   (gap 최소 60.12mm · 45.47mm, 물체 41mm 대비)
```

**우리 선별기는 시각 특징 실패를 볼 수 없다.** 동기화·IMU·길이는 보지만
"SLAM 이 붙을 만한 화면인가"는 못 본다. → LIMITS 등재.

**수집 계획 계수: 필요한 편수 N 이면 N/0.88 편을 찍는다.**

---

## 8. `ai → dev` 병합

MR 병합 완료. `dev` 에 BE 1차가 들어가 있고 BE 2차가 남아 있다.
push·MR 은 프록시 403 으로 사용자가 직접 (`AI/tools/push_both.sh`).

---

## 9. 실데이터 첫 정책 — 학습·평가 🟢 (단, n=1)

도경 축수정본 74편(89.87° → 0.11° 턱축 수정 반영). train 60 / holdout 14.

```
trans_mm   chunk 21.996   identity 71.422   개선 69.20%
rot_deg    chunk  3.359   identity  6.024   개선 44.24%
gap_mm     chunk  2.095   identity  3.311   개선 36.73%

horizon 별 trans_mm
[2.0988, 6.6641, 12.8851, 19.3639, 25.8558, 31.5251, 36.6727, 40.9030]
```

측정 조건: 분할시드 42 · 60 epoch · `action_horizon=8` · `obs_down_sample_steps=1` ·
V100 · holdout 14편.

**E2 의 회전 개선은 2.9% 였다.** 44.2% 로 뛴 것은 도경의 턱축 수정 폭과 방향이 맞는다.
다만 **데이터셋도 같이 바뀌었으므로 축 수정 단독 기여로 귀속할 수 없다.**

### 학습 소요

```
v4  (76편 · 16,328 프레임 · down_sample 3)   60 epoch  2h19m  최종 loss 0.013
v10 (60편 ·  3,898 프레임 · down_sample 1)   60 epoch  ~30m   epoch 55 loss 0.017
```

---

## 10. 착수했고 아직 안 끝난 것 — 5-fold A/B ⚠️ 미검증

`AI/tools/run_folds_0918.sh` (해시 `51da2d3023b9d37548498c61874b1925`).

```
분할시드   42 ~ 46
조건 A     warm start 없음
조건 B     --init-checkpoint outputs/e2_C/checkpoints/latest.ckpt
공통       60 epoch · batch 8 · GPU 1/2/3/4/6
           --override task.action_horizon=8 --override task.obs_down_sample_steps=1
```

**정답 아는 행을 앞에 뒀다** — 분할시드 42 를 다시 변환해 60/3898 · 14/816 이
재현되는지 먼저 본다. 안 맞으면 조건이 달라진 것이므로 `exit 3` 으로 죽는다.

```
outputs/ds_f0918_42_train.provenance.json    편 60/60  프레임 3898/3898
outputs/ds_f0918_42_holdout.provenance.json  편 14/14  프레임  816/816
재현 OK
```

판정선: **fold 5개의 B−A 부호가 전부 같으면** 시뮬 사전학습이 축수정본에서도 돕는다.
부호가 갈리면 "효과 없음"이 아니라 **n=5 로도 못 가른다** 는 뜻이다 —
게이트를 낮추지 말고 실험 개수를 줄인다.

---

## 11. 미해결 · 상대 트랙 대기

| 무엇 | 누구 |
|---|---|
| `pass_90.txt` 전달 — 도경 이번 묶음은 74편 기준본이다. 우리 81편과 교차 필요 | 트랙 B → A |
| `trajectory_validation.json` 의 상태 판정 기준 | 도경 |
| `applied_imu_shift_ms` −15.11 (92편 전부 동일) — 실측인가 고정값인가 | 도경 |
| `v4.zarr` 에 든 76편이 어느 편들인가 | 현석 |
| `profileKey`/`algorithm` enum 문자열 · `actionSpace` 에 `EEF_RELATIVE_ROT6D` 수용 여부 | 은찬 |
| gripper.csv 게이트에 `detected > 0` 추가 | 은찬 |
| Jetson 보드 확정 → `frameworkVersion` · `contractVersion` | HW/PM |
| 1a·1b 실물 재생 안전 절차 승인 (간섭·정렬·속도·비상정지) | HW · 트랙 A |

---

## 12. 어시스턴트 오류 18건

`TS_assistant_errors_0918_2.md` 로 분리. 요약만:
**인용 4 · 전달미확인 2 · 판별력0 검사 3 · 코드·실행 5 · 판단 4.**

세션 중에 "다섯 건"이라 말했고 KPT 초판에 11건으로 적었다. 다시 세니 18건이다.
**5 → 11 → 18.** 오류 개수 자체를 모수 없이 세 번 틀렸다.

이 문서 초판의 "모터 고장" 서술도 같은 계열의 오류다 — **프로젝트 지침의 낡은 줄을
D-AI-58 보다 우선해 읽었다.** 위 정정 블록으로 남긴다.

---

## 13. 오늘 커밋된 계측기

```
AI/tools/check_collection_quality.py   수집 선별 (게이트/관측 분리)
AI/tools/check_gap_validity.py         3b06dd90b9fcd2c14d5eb019e58a8233  자체검증 10/10
AI/tools/probe_ckpt_contract.py        자체검증 6/6
AI/tools/check_zarr_parity.py          b337f995d068a840b6fbaf080d5c77ae  자체검증 10/10
AI/tools/summarize_slam_batch.py       81fdfc2206d73bdee3fb20ae765c31bb
AI/tools/subset_umi_zarr.py            자체검증 5/5
AI/tools/run_folds_0918.sh             51da2d3023b9d37548498c61874b1925
AI/configs/real/umi_s22_canonical_pinch_side_grasp_provisional.json
                                       a7b492aa42e9d6744b22b174271f27d1
```

마지막 config 는 `MEASURE_handeye_from_video_0912.md` 로부터 재구성한 **잠정본**이다.
X축 회전 105.0089° · t = (-0.0014, -0.0401, +0.1497) m.
**마커 중심 대 손끝 26.6mm 모호성이 미해결**이며 파일 안에 `_UNRESOLVED` 로 적어뒀다.
