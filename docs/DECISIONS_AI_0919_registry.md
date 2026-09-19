# DECISIONS — 2026-09-19 · 스킬 레지스트리를 실측에 맞춘다

작성자: 김준태(트랙 B) · 도구 `AI/tools/update_skill_registry_0919.py` (자체검증 6/6, 멱등 확인)

---

## D-AI-63 · 스킬 레지스트리 5건을 현행 실측·계약으로 갱신

**작성자**: 김준태(트랙 B) · **상태**: 확정 (트랙 B 소유 파일) · **FE·BE 통보 필요**

레지스트리는 0902 작성본이었고, 이후 실측과 어긋난 값을 다섯 파일이 똑같이 들고 있었다.
이 파일에서 **프런트 목록 · BE 계약 · VLM 출력 스키마**가 파생되므로 낡은 채로 두면
조용히 틀린다.

| 필드 | 이전 | 이후 | 근거 |
|---|---|---|---|
| `robot.dof` | 6 | **5** | 실측 5자유도 + 그리퍼 🟢 |
| `policy.action_space` | `joint_delta` | **`eef_relative_rot6d`** | D-AI-60 · 현행 출력 (8,10) |
| `policy.ckpt_uri` | 빈 문자열 | `outputs/f0918_B_42/checkpoints/latest.ckpt` | 배포본 sha256 `abb8a77d…` 🟢 |
| `policy.trained_on` | 빈 문자열 | v10 74편 중 train 60편(분할시드 42) · warm start e2_C | MEASURE_folds_real_0919 |
| `policy.contract_version` | `0.1.0-provisional` | `umi_official/0.1.0` | 공식 UMI zarr 포맷 |
| `workspace_m.x` | [0.10, 0.25] | **[0.34, 0.46]** + 조건 문자열 | 도달 포락선 실측(측면 파지·base_height 0) 🟢 |
| `object_size_mm` | [15.0, 25.0] | **`[]`** | 기획서 "1.5~2.5cm" 는 근거 불분명(D-AI-57). 대상물 미확정 |

### 왜 `object_size_mm` 을 지우고 비워 두는가

`validate_entry` 가 *"object_size_mm 은 [min, max] 두 값이어야 한다"* 로 **명시적으로
걸어준다.** 옛 값을 남기면 미확정이 확정처럼 보이고, `None` 은 로더를 깨뜨린다.
**"없음"과 "괜찮음"이 같은 출력으로 나오지 않게** 빈 값 + 검증 실패로 남긴다.
HW 가 5스킬 대상물을 확정하면 채운다.

### 바꾸지 않은 것

**`status` 는 5개 전부 `planned` 로 남긴다.** 놓기 스크립트가 0/5 이고 실물 롤아웃이 0건이다.
`gate` 도 `null` 로 남긴다 — 배포 ckpt 는 실 시연 데이터 학습본이고 **롤아웃 수치가 없다.**

갱신 후 정직한 숫자 🟢:
```
스킬 5개 · 서로 다른 학습 정책 1개
  상태: planned 5
  → 실제로 동작하는 스킬은 0개다
  → 안 A 상태다. '5개를 학습시켰다'가 아니라 '파지 정책 1개를 5개 작업이 공유한다'
```

### 연쇄 영향
```
FE   클릭이 보내는 skill_id 5개는 그대로 (contract/ids.py 변경 없음)
BE   actionSpace 에 EEF_RELATIVE_ROT6D 가 필요하다는 근거가 레지스트리에도 박혔다 (D-AI-61 2번)
트랙 A  없음
```

### 되돌릴 조건
- 실물 롤아웃 수치가 나오면 `gate` 와 `status` 를 그 값으로 채운다
- 대상물이 확정되면 `object_size_mm` 을 채우고, **물체가 바뀌면 전문가 코드를 새로 쓴다**
- `workspace_m` 은 base_height 가 바뀌거나 베이스 정렬이 적용되면 재측정 대상이다

### 🟡 미검증
`check_skills.py` 전체 검사는 MuJoCo 가 필요해 로컬에서 못 돌렸다.
`shared_policy_report` 와 로더는 통과했고, **전체 검사는 서버에서 1회 돌려야 한다.**
