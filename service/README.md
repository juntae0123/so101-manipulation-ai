# 촬영 게이트 서비스 — BE 연동 계약

작성 김준태 (트랙 B) · 2026-09-21 · `gate_api.py` (자체검증 12/12, 의존성 **표준 라이브러리만**)

## 무엇을 푸는가

촬영 게이트 15항목은 **전부 메타데이터만으로** 판정된다. 영상이 필요 없다.
그래서 **영상을 올리기 전에** 판정할 수 있고, 사람이 아직 현장에 있을 때
"다시 찍으세요"가 뜬다. SLAM 을 돌리고 나서 알면 늦다.

```
앱        촬영 → manifest.json + frames.csv + IMU csv 업로드 (수백 KB)
BE        POST /gate/capture  → 1초 이내 판정
앱        verdict != PASS 면 재촬영 유도 (사유는 rows 에 있다)
          PASS 면 그때 영상 업로드
BE        sha256 무결성 · 중복 제거 · 학습 서버 큐 · 판정 이력
학습서버  SLAM → SLAM게이트 → plan → zarr → zarr게이트 → 학습
```

**게이트 로직은 여기 한 곳에만 둔다.** 앱(Kotlin)에 다시 구현하면 파이썬 판정과
갈리고, 그 순간 같은 데이터에 두 개의 답이 생긴다. 2026-09-21 하루에만 기준을
세 번 고쳤다 — 구현이 둘이면 그때마다 두 번 고쳐야 한다.

## 띄우기

```bash
python3 AI/service/gate_api.py --selftest            # 먼저 이게 12/12 여야 한다
python3 AI/service/gate_api.py --serve --port 8971
```

파이썬 3.10+ 만 있으면 된다. FastAPI·numpy 불필요. 서비스는 뜰 때 자체검증을
먼저 돌리고, 실패하면 **뜨지 않는다.**

## API

### `POST /gate/capture`

```json
{
  "episode": "rec_1789701868830_2c55035d",
  "video_bytes": 16756474,
  "files": {
    "manifest.json": "<파일 내용 그대로>",
    "frames.csv": "...",
    "accelerometer.csv": "...",
    "gyroscope.csv": "...",
    "encoded.csv": "..."
  }
}
```

| 필드 | 필수 | 비고 |
|---|---|---|
| `files["manifest.json"]` | ✅ | |
| `files["frames.csv"]` | ✅ | |
| `files["accelerometer.csv"]` `["gyroscope.csv"]` | 권장 | 없으면 IMU 4항목이 **미판정** |
| `files["encoded.csv"]` | 권장 | 없으면 pts 시각차가 **미판정** |
| `video_bytes` | 권장 | **안 보내면 "영상 파일" 항목이 미판정.** 영상 자체는 안 보낸다 |

응답 200:

```json
{
  "episode": "...", "gate_version": "capture-gate-v2",
  "verdict": "PASS|FAIL|INCOMPLETE",
  "passed": 15, "failed": 0, "unknown": 0, "total": 15,
  "soft_unmet": 0, "soft_total": 1,
  "rows": [{"name": "프레임 수 하한", "ok": true, "got": "100", "want": ">= 60", "soft": false}],
  "measured": { "frames": 100, "fps_median": 30.0, "...": "..." }
}
```

응답 400: `{"error": "...", "missing": ["manifest.json"]}`

### `GET /gate/spec`
거는 기준값 전부 + `spec_sha256`. 앱이 "왜 떨어졌나"를 사용자에게 설명할 때 쓴다.

### `GET /health`
살아있음 + `spec_sha256`. **가용성 확인이지 기능 확인이 아니다** — 기능은
`/gate/capture` 에 정상 편 하나를 넣어 PASS 가 나오는지로 확인한다.

## BE 가 반드시 지킬 것

1. **`verdict != "PASS"` 면 영상을 올리지 않는다.** `INCOMPLETE`(미판정)는 통과가 아니다
2. `total` 은 **필수 항목 수**다. `soft_unmet` 은 권고이고 **폐기 사유가 아니다** —
   길이가 긴 시연은 난이도 경고이지 불량이 아니다 (결과 모방 기조)
3. `rows` 를 그대로 저장한다. 나중에 "왜 떨어졌나"를 집계하면 **수용률과 폐기 사유
   분포가 저절로 나온다** (S15P21A103-113 완료 기준이 제품 기능이 된다)
4. `gate_version` 을 같이 저장한다. 기준이 바뀌면 옛 판정과 섞이면 안 된다

## 실측 참고 🟢

현석 raw 92편에 이 게이트를 건 결과 (2026-09-21):

```
통과 89 · 불합격 3 · 미판정 0 / 전체 92
불합격 사유: 촬영 성공 표기 3편 (그중 2편은 프레임 수 하한 미달)
권고 미충족: 89편 (길이) — 폐기 사유 아님
```

**필수 불합격 3.3%.** 이 비율이 다른 배치에서도 같은지는 미검증.

## 한계 — 먼저 말한다

- 게이트는 **데이터 품질만** 본다. 통과한 데이터로 학습한 정책이 좋은지는 말하지 않는다
- 영상 자체는 검사하지 않는다 (모션블러·가림은 SLAM 게이트에서 잡힌다)
- BE 실행 환경에서 파이썬을 띄울 수 있는지 **미확인** — 안 되면 이 서비스를 학습
  서버에 두고 BE 가 HTTP 로 호출한다. 어느 쪽이든 구현은 하나다
- 인증 없음. 사내망 전용으로 두거나 BE 가 앞단에서 막는다
