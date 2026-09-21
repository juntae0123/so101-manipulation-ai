# AI/notebooks

**저장소의 노트북은 템플릿이다. 실행은 복사본에서 한다.**

주피터가 실행 출력을 파일에 저장하므로, 저장소 노트북을 직접 실행하면
다음 `git pull` 이 "local changes would be overwritten" 으로 막힌다 (2026-09-21 실증).

```bash
# [서버]
cd ~/S15P21A103 && git pull origin ai
cd AI/notebooks && cp 01_slam_to_ckpt.ipynb work_$(date +%m%d).ipynb
# 주피터에서는 work_*.ipynb 를 연다
```

`work_*.ipynb` 는 `.gitignore` 에 있다. 템플릿이 갱신되면 복사만 다시 한다.
템플릿을 고쳐야 하면 로컬에서 고쳐 커밋하고, 서버는 pull 만 받는다.

## 01_slam_to_ckpt.ipynb

촬영 → 촬영게이트 → (SLAM: 현석 WSL) → SLAM게이트 → plan → zarr → zarr게이트 → 학습 → ckpt검사.
0번 셀만 고치면 된다. 각 셀은 하위 프로세스 출력을 줄 단위로 실시간 표시한다.
