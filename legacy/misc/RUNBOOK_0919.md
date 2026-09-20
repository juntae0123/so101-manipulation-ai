# 실행 순서 — 2026-09-19 자력 해소 3종

전부 **한 줄 명령**이다. 그대로 복붙한다.
`~/S15P21A103` 이 서버 저장소 경로다 (`run_folds_0918.sh` 가 이미 그렇게 쓴다 🟢).

---

## 0. 올리고 받는다

```
# [로컬]
bash AI/tools/push_both.sh
```

```
# [서버]
cd ~/S15P21A103 && git pull
```

---

## 1. 자체검증 3종 (30초. 실패하면 아래를 돌리지 않는다)

```
# [서버]
cd ~/handoff && for t in solve_grasp_anchor sweep_camera_mount probe_empty_scene_refusal; do echo "== $t"; ~/envs/handoff312/bin/python ~/S15P21A103/AI/tools/$t.py --selftest | tail -1; done
```

기대: `8 / 8`, `11 / 11`, `11 / 11`.

---

## 2. 경로 확인 — 가정하지 않는다

```
# [서버]
ls -d ~/handoff/outputs/e1_s* ~/handoff/outputs/demos_* ~/S15P21A103_umi/AI/datasets/umi_real_relative_20260918_v10_orbslam_cadtcp_video_aligned_provisional_74ep 2>&1
```

없는 게 나오면 아래 명령의 그 경로만 실제 이름으로 바꾼다.
**있는 줄 알고 돌리면 "없음"과 "괜찮음"이 같은 실패로 나온다.**

---

## 3. 카메라 마운트 허용치 스윕 (약 23분, GPU 5장)

```
# [서버]
cd ~/handoff && ~/envs/handoff312/bin/python ~/S15P21A103/AI/tools/sweep_camera_mount.py --plan --handoff ~/handoff --checkpoint ~/handoff/outputs/e1_s0/checkpoints/latest.ckpt
```

출력 맨 아래의 `nohup ...` 5줄을 그대로 복붙해 띄운다. 그 다음:

```
# [서버]
grep -h SWEEPCAM_GPU ~/handoff/outputs/sweep_camera/gpu*.log | wc -l
```

`5` 가 나오면 끝. 집계:

```
# [서버]
cd ~/handoff && ~/envs/handoff312/bin/python ~/S15P21A103/AI/tools/sweep_camera_mount.py --collect --handoff ~/handoff
```

**판별 조건(45도)이 기준선보다 안 떨어지면 도구가 스스로 죽는다.** 그건 결과가 아니라
`camera_path` 가 안 읽힌다는 신호다. 그 경우 사양을 인용하지 마라.

---

## 4. 파지 배치 solver (격자 600 × 74편. 처음엔 작게)

먼저 작게 돌려 시간을 잰다.

```
# [서버]
cd ~/handoff && ~/envs/handoff312/bin/python ~/S15P21A103/AI/tools/solve_grasp_anchor.py --dataset ~/S15P21A103_umi/AI/datasets/umi_real_relative_20260918_v10_orbslam_cadtcp_video_aligned_provisional_74ep --task configs/can_side.yaml --reference-demos ~/handoff/outputs/demos_6000 --limit 10 --nx 3 --ny 3 --nz 2 --nyaw 4 --out ~/handoff/outputs/grasp_anchor_pilot.json
```

납득되면 전체:

```
# [서버]
cd ~/handoff && nohup ~/envs/handoff312/bin/python ~/S15P21A103/AI/tools/solve_grasp_anchor.py --dataset ~/S15P21A103_umi/AI/datasets/umi_real_relative_20260918_v10_orbslam_cadtcp_video_aligned_provisional_74ep --task configs/can_side.yaml --reference-demos ~/handoff/outputs/demos_6000 --out ~/handoff/outputs/grasp_anchor.json > ~/handoff/outputs/grasp_anchor.log 2>&1 &
```

---

## 5. 빈 장면 거부 (프레임 찍는 게 전부다)

폰을 삼각대에 고정하고 **카메라를 건드리지 않은 채** 두 번 찍는다.
책상 위 대상물만 빼고 나머지는 그대로 둔다. 조건당 30장.

```
# [서버]
mkdir -p ~/frames_with ~/frames_without
```

Jupyter 로 업로드한 뒤:

```
# [서버]
cd ~/handoff && ~/envs/handoff312/bin/python ~/S15P21A103/AI/tools/probe_empty_scene_refusal.py --ckpt ~/handoff/deploy/so101_pick_v1.ckpt --task configs/can_side.yaml --with-dir ~/frames_with --without-dir ~/frames_without --device cuda:1 --out ~/handoff/outputs/empty_scene.json
```

사전등록: `AI/docs/PREREG_empty_scene_refusal_0919.md`
**예측은 A 미달(차이 0~5mm) 쪽에 걸어놨다.** 빗나가면 방향까지 기록한다.

---

## 되돌리기

```
# [서버]
rm -rf ~/handoff/configs/_sweep_camera ~/handoff/outputs/sweep_camera ~/handoff/outputs/grasp_anchor*.json ~/handoff/outputs/empty_scene.json
```

기존 설정·데이터·체크포인트는 어느 단계에서도 안 건드린다.
