# vendor — 원본 무수정 사본

| 파일 | 원본 | sha256 | 복사일 |
|---|---|---|---|
| shanks_kinematics.py | 김현석 handoff 2026-09-18 `control/shanks/kinematics.py` | `1371ed3f7b4708d182e2e5cc7985984b857ac5468d68bfb0ce6cbb09f11a4ba7` | 2026-09-19 |

**수정 금지.** 원본이 바뀌면 다시 복사하고 해시를 갱신한다.
so101_ver1_original.urdf 관절 체인 FK/IK. 순수 numpy, 드라이버 불필요.
자체검사: `python3 shanks_kinematics.py` → `KINEMATICS_SELF_CHECK_OK`
