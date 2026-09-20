# AI/legacy — 안 쓰는 것을 여기 둔다 (2026-09-21)

**지운 게 아니라 옮긴 것이다.** 기록과 재현을 위해 남긴다.

## tools_bc_dagger/

구 경로(BC · ACT · DAgger, ~2026-09-15) 스크립트 17개.
현행 스택(handoff · MuJoCo 3.13.0 · 공식 UMI diffusion)과 무관하다.

⚠️ **구 경로 수치를 현행 스택 예측의 근거로 쓰지 마라.**
2026-09-17 실증 — BC 77.7% 를 앵커로 E1 을 65~85% 로 예측했는데 실제는 97.0% 였다.
아키텍처·표현·데이터가 전부 다르다.

## misc/

- `RUNBOOK_0919.md` — `AI/tools/` 에 잘못 놓여 있었다. `AI/docs/RUNBOOK_real_0921.md` 로 대체
- `check_real_traj_ik.py.bak_joint_margin` — 패치 적용 전 백업
- `SHARE_status_briefing_0919-1.md` — 중복본
- `__pycache__.*` — 바이트코드

## 판정 기준 (2026-09-21)

저장소 추적 파일 446개를 본문까지 읽어 `AI/tools/` 134개의 참조를 세었다.
**참조 0건 20개.** 거기에 "구 경로인가"를 겹쳐서 17개만 옮겼다.

**참조 0건이 곧 안 쓰는 것은 아니다.** 남긴 것:
- `patch_joint_margin.py` — 2026-09-20 작성, 현행
- `check_determinism.py` — 수동 실행 도구라 참조가 없는 게 정상

## 문서는 옮기지 않았다

`AI/docs/` 197개는 그대로 둔다. DECISIONS·MEASURE·TS 는 기록이고,
옮기면 나중에 찾기 어려워진다. **기록은 자리를 지킨다.**
