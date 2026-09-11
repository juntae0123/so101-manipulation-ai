#!/usr/bin/env bash
# Push to the team remote and, if configured, to a personal mirror in one step.
# 팀 원격과, 설정돼 있으면 개인 미러에 한 번에 올린다.
#
# The mirror is a subtree split of AI/, so its history is the AI-part commits and
# nothing else. `subtree split` recomputes hashes every time, so the mirror push
# is a force push -- that repository holds nothing the split does not regenerate.
# 미러는 AI/ 의 subtree split 이라 히스토리가 AI 파트 커밋만으로 이뤄진다.
# `subtree split` 은 매번 해시를 새로 계산하므로 미러 push 는 force 다. 그 저장소에는
# split 이 다시 만들어내지 못하는 것이 없다.
#
# 사용법:  bash AI/tools/push_both.sh          (저장소 어디서든)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"      # 팀 GitLab 쪽 브랜치 (ai)
MIRROR_LOCAL="ai-standalone"                    # split 결과를 담을 로컬 브랜치
MIRROR_REMOTE="main"                             # 개인 미러의 기본 브랜치

# Only tracked changes block the push. Untracked files are not going anywhere --
# refusing because of one is refusing for a reason that does not exist.
# push 를 막는 것은 추적 중인 변경뿐이다. 추적되지 않는 파일은 어차피 올라가지
# 않으므로, 그것 때문에 거부하는 것은 존재하지 않는 이유로 거부하는 것이다.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "✗ 커밋되지 않은 변경이 있다. 커밋하고 다시 실행하라:"
  git status --short --untracked-files=no
  exit 1
fi

UNTRACKED="$(git ls-files --others --exclude-standard | head -5)"
if [ -n "${UNTRACKED}" ]; then
  echo "· 추적되지 않는 파일이 있다 (올라가지 않는다):"
  echo "${UNTRACKED}" | sed 's/^/    /'
  echo
fi

echo "=============================================="
echo " 1/2  팀 GitLab (origin ${BRANCH})"
echo "=============================================="
git pull --rebase origin "${BRANCH}"
git push origin "${BRANCH}"

if ! git remote get-url gh >/dev/null 2>&1; then
  echo
  echo "· 리모트 'gh' 가 없다. 개인 미러는 건너뛴다."
  echo "   추가하려면: git remote add gh <개인 미러 저장소 URL>"
  exit 0
fi

echo
echo "=============================================="
echo " 2/2  개인 미러 (gh ${MIRROR_REMOTE}) — AI/ 만"
echo "=============================================="
git branch -D "${MIRROR_LOCAL}" >/dev/null 2>&1 || true
git subtree split --prefix=AI -b "${MIRROR_LOCAL}" -q
echo "AI/ 커밋 $(git rev-list --count "${MIRROR_LOCAL}") 개"

# Strip assistant trailers from the mirror's messages (2026-09-11).
# 미러 메시지에서 어시스턴트 서명 줄을 지운다.
#
# 팀 GitLab 쪽은 못 지운다 — 공유 브랜치이고 65커밋이 이미 dev 에 머지됐다.
# 미러는 다르다: 매번 subtree split 으로 **다시 만들어** force push 하므로
# 여기서 지우면 과거 커밋까지 전부 깨끗해지고, 다음 push 에서도 그대로 유지된다.
# 조율 비용 0. `claude/규칙_커밋_서명금지.md`
#
# 던지는 대상은 throwaway 브랜치 ai-standalone 뿐이다. ai 는 건드리지 않는다.
echo "· 미러 메시지에서 어시스턴트 서명 줄 제거"
git update-ref -d "refs/original/refs/heads/${MIRROR_LOCAL}" 2>/dev/null || true
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
  --msg-filter "sed -E '/^Co-Authored-By: Claude/d; /^Claude-Session:/d; /Generated with \[Claude Code\]/d'" \
  -- "${MIRROR_LOCAL}" >/dev/null
git update-ref -d "refs/original/refs/heads/${MIRROR_LOCAL}" 2>/dev/null || true
rm -rf "$(git rev-parse --git-dir)/refs/original" 2>/dev/null || true

LEFT="$(git log "${MIRROR_LOCAL}" --format='%h' \
  --grep='Co-Authored-By: Claude' --grep='Claude-Session:' -i -- | wc -l | tr -d ' ')"
if [ "${LEFT}" != "0" ]; then
  echo "✗ 미러에 서명이 남은 커밋 ${LEFT}건. push 하지 않는다."
  exit 1
fi
echo "  남은 서명 커밋 0건 확인"

git push gh "${MIRROR_LOCAL}:${MIRROR_REMOTE}" --force

echo
echo "완료. origin/${BRANCH} · gh/${MIRROR_REMOTE} 최신."
