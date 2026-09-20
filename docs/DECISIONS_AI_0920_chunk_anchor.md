# D-AI-80 — 청크 액션은 청크 시작 pose 기준(앵커)이다

작성자: 김준태 (트랙 B) · 2026-09-20 · 근거 `AI/docs/MEASURE_chunk_anchor_0920.md`

## 결정

`umi_relative_chunk/0.2.x` 의 `action[i, k]` 는 **전부 `chain[i]`(청크 시작 pose) 기준**이다.
청크 안에서 누적하지 않는다. 절대 궤적은 `P_k = T_anchor @ A[k]` 로 편다.
앵커는 **청크 사이에서만** 넘어간다 (실물에서는 재관측 pose).

## 왜

v10 실데이터 **전수 74편 · 비교 17672건**에서 앵커 오차 중앙·최대 모두 **0.0000 mm**,
누적 74.7 mm, 기존 배포 공식 55.0 mm. k에 따라 0 → 42 → 105 → 187 mm 로 증가.
(1차 20편 5648건도 같은 결론 — 표본을 3.7배로 늘려도 앵커는 최대까지 0 이다)

## 되돌릴 조건

- 다른 수집 배치(현석 `s22_pick_v3_atlas_v1` 등)에서 앵커 오차가 0 이 아니게 나오면
- 트랙 A 가 v10 생성 코드에서 누적 규약으로 쓴다는 것을 소스로 보이면
- 실물에서 앵커로 고쳤는데도 같은 원호가 나오면 (그때는 원인이 다른 곳이다)

## 영향

- 수정: `AI/deploy/so101_infer.py::unroll`
- 무영향(이미 앵커): `policy_to_joints.decode_chunk` · `convert_v10_to_umi` · `check_real_traj_ik` · `run_policy_realtime`
- 트랙 A 영향: **없음** (데이터 포맷 변경이 아니라 해석 확정이다)
- 매니페스트에 `chunk_anchor: "chunk_start"` 추가. 구 매니페스트는 `smoke_deploy_ckpt` 가 실패로 떨어뜨린다 (689e1dc)
