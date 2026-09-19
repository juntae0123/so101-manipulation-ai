# TS — 2026-09-19 어시스턴트 오류

## 1. 스윕 잡에 `MUJOCO_GL=egl` 이 없어 5개 GPU 잡이 전부 즉사

### 증상
```
mujoco.FatalError: an OpenGL platform library has not been loaded into this process
GLFWError: b'X11: The DISPLAY environment variable is missing'
→ 5개 잡 전부 exit 1 (첫 조건 렌더 시점)
```

### 어떻게 좁혔나
`set -eu` 라 첫 실패에서 죽는다. 그래서 exit 가 **즉시** 났고, 20편 중간이 아니라
**첫 관측 렌더**에서 죽었다는 것이 시간만으로 좁혀졌다. 로그 마지막 프레임이
`env.py:114 mujoco.Renderer(...)` 였다.

### 진짜 원인 — 내가 대조를 절반만 했다
지침에 *"일괄 실행 스크립트의 명령은 전부 `--help` 또는 소스로 인자를 대조한 뒤 쓴다"*
가 있다. 나는 `evaluate.py` 의 **argparse 를 소스로 대조했고**, 그걸 근거로
"인자 대조 🟢" 이라고 문서와 docstring에 적었다. **환경변수는 안 봤다.**

저장소의 기존 러너 8종이 전부 이 줄을 갖고 있었다 —
`run_e1_train.sh:61` · `run_e2.sh:22` · `run_e2_folds.sh:82` ·
`run_dagger_campaign.sh:19` · `run_chunking_campaign.sh:19` · `check_reproducibility.sh:22` …
**한 번만 grep 했으면 나왔다.**

### 왜 자체검증이 못 잡았나
자체검증 11행이 전부 **기하와 조건표**만 봤다. 잡 파일의 내용을 한 줄도 검사하지
않았다. 생성물을 안 보는 계측기였다.

### 조치
- `JOB_HEADER` 로 `export MUJOCO_GL=egl` + `cd` + 사전 확인을 묶었다
- 자체검증 3행 추가 (11 → 14): 헤더에 그 줄이 있는가 · 빼면 걸리는가 · cd 하는가
- 생성된 잡 파일에 `bash -n` 구문 검사까지 돌려 확인 🟢

### 일반화
**인자 대조 ≠ 실행 환경 대조.** 다음부터 러너를 새로 쓸 때는
`grep -rn "export \|^[A-Z_]*=" 기존_러너` 로 **환경변수를 먼저 훔쳐본다.**
기존에 돌던 스크립트가 있으면 그게 사양이다.
