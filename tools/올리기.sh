#!/usr/bin/env bash
# 어제~오늘 작업물을 팀 GitLab + 개인 GitHub 미러에 한 번에 올린다.
# 실행:  bash AI/tools/올리기.sh      (저장소 어디서든)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "== 1/3  올릴 것 =="
git add -A AI/docs AI/tools AI/configs AI/requirements-sim-server.txt 2>/dev/null || true
git status --short --untracked-files=no | sed 's/^/   /'
if git diff --cached --quiet; then
  echo "   (변경 없음 — 이미 올라가 있다)"; exit 0
fi

echo
echo "== 2/3  커밋 =="
git commit -q -m "feat: 3일 과제 ②③ 완료, 실 시연 실행가능성 계측기 (S15P21A103-113, D-AI-49, D-AI-50, D-AI-51)

- E1 시뮬 UMI 모방학습 291/300 = 97.0% [94.4, 98.4]
- E3 정책 2종 자기 96~97% / 교차 1.3%, 게이트 4개 통과
- 실 시연 파지 도달성 99.6% (앵커 정정). 궤적 기준 29.2% 는 폐기율 아님
- check_real_traj_ik.py: 연속 IK + 자체검증 4종 + 앵커/구간 선택
- handoff 패치 3종(seed/override/lift) 전부 멱등"
git --no-pager log -1 --oneline | sed 's/^/   /'

echo
echo "== 3/3  push (팀 GitLab + 개인 GitHub) =="
bash AI/tools/push_both.sh
