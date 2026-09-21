# TS — 어시스턴트 오류 (2026-09-21)

"고쳤다"가 아니라 "어떻게 좁혔나"를 적는다.

## 1. 파일 내용을 재타이핑해서 1글자를 망가뜨렸다

**증상** 클라우드에서 만든 `MEASURE_trim_default_0921.md` 를 로컬 저장소에 옮기려고
base64 문자열(5068자)을 **출력에서 읽어 heredoc 에 다시 찍었다.**
길이는 3801바이트로 같은데 sha256 이 달랐다.

**좁힌 방법** 200바이트 블록마다 md5 를 찍어 양쪽을 나란히 놓았다.
1600·2000 블록은 같고 **1800 블록만 달랐다** → 그 구간만 `repr` 로 떠서 비교.

```
클라우드  ... 안 나온다.
로컬      ... 안 나오다.       <- \xec\x98\xa8(온) 이 \xec\x98\xa4(오) 로
```

base64 한 글자가 바뀌면서 UTF-8 한 바이트가 바뀌었다. **길이는 그대로였다.**

**원인** 이 환경의 지침이 명시한 금지 — *"도구 출력에서 파일 내용을 다시 찍어 쓰지
않는다(잘렸을 수 있다)"* — 를 어겼다. 길이가 같아서 `wc -c` 로는 안 잡혔다.

**조치** 파일 이동은 `device_commit_files` 로 하고, **옮긴 뒤 sha256 을 대조한다.**
`efa5adf7…` 일치 확인 후에야 커밋했다. 크기 비교는 검사가 아니다.

## 2. 학습 ckpt 를 배포 검사기에 바로 넣었다 (export 단계 누락)

**증상** 노트북 9번 셀이 `KeyError: 'actionSpec'` 로 죽었고, 그 앞의
`state_dicts 는 ema_model 뿐` 항목이 `['model','ema_model']` 로 불합격이 떴다.

**진짜 원인** ckpt 도 검사기도 멀쩡했다. **노트북이 단계를 하나 빼먹었다.**
학습기가 내는 `manifest.json` 은 실행 기록(run_id·loss·sha256)이고,
배포 계약(`actionSpec.horizon` · `nParams` · `sha256_export` · `chunk_anchor`)을
만드는 것은 `AI/tools/export_deploy_ckpt.py` 다. 학습 ckpt 는 `model` 과 `ema_model`
을 둘 다 갖는 게 정상이고, `ema_model` 만 남기는 것도 export 의 일이다.

**조치**
- 노트북 9번 셀을 **export -> 검사** 두 단계로 바꿨다 (`--out <파일>.ckpt`,
  매니페스트는 `out.with_suffix(".manifest.json")` — 소스로 인자 대조함)
- `smoke_deploy_ckpt.py` 가 **죽지 않게** 고쳤다. `manifest_contract()` 가 없는 키를
  모아 돌려주고, 검사기는 `없는 키 5 / 전체 5 — 학습 매니페스트로 보인다` 로 보고한다.
  **죽는 검사기는 검사기가 아니다.** 자체검증 6 -> 9 행 (학습 매니페스트 / 배포
  매니페스트 / 일부만 있는 매니페스트 세 갈래를 판별하는지)

## 3. 8 epoch 결과를 데이터 근거로 쓸 뻔했다

`beats_hold_baseline: false` (0.52x) 를 보고 trim 3.0 데이터 탓이라고 쓸 뻔했다.
**교락이 둘이다.** epoch 8 (참조 3.52x 는 120 epoch) 과 trim 3.0.
train_loss 가 1.09 -> 0.231 로 단조 감소 중이었다 — 수렴 전이다.
분리하려면 같은 epoch 에서 trim 만 바꾼 짝이 필요하다 (PREREG_trim_epochs_0921).
