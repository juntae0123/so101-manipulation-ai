"""The decoded, robot-frame intermediate between the app's raw bundle and the contract.
앱의 raw 번들과 계약 사이에 있는 **디코딩·로봇좌표 중간 표현**.

## ⚠️ 여기는 "raw" 가 두 개라는 오해가 나기 쉬운 자리다 — 2026-09-08 정정

보관용 raw 는 이 파일이 아니다. **`docs/SPEC_umi_raw_bundle.md` 의
`schema: "umi_raw/0.1.0"` 번들**(ZIP: `episode.json` · `poses.csv` · `imu.csv` ·
`frames.csv` · `gripper.csv` · `frames/*.jpg`)이 보관용이고, jpg 재인코딩이 금지된
무손실 원본이다. 그게 바뀌면 양 트랙 합의 + D- 기록이 필요하다.

이 파일은 그 번들을 **디코딩하고 로봇 베이스 좌표로 옮긴 중간 표현**이다.
`track_a/convert/arcore.py` 가 번들 → 이것으로 만들고, `umi/convert.py` 가
이것 → 계약으로 만든다.

    umi_raw/0.1.0 번들   →   RawEpisode (이 파일)   →   contract.Episode
    (보관용, SPEC 정본)      (중간, 로봇 좌표)          (학습용)

처음 이 파일의 버전을 `"0.1.0-provisional"` 로 적었다. 번들 스키마와 이름도
번호도 같아서 두 벌로 읽혔다. `umi_intermediate/…` 로 바꿨다.

## 세 가지 설계 결정과 그 이유

**1. pose 와 이미지의 길이를 맞추지 않는다.**
`pose_timestamp` 는 (T,), `image_timestamp[cam]` 은 (Ti,) 로 따로 둔다. ARCore 는
pose 와 RGB 가 별도 스트림이고 프레임 수가 같다는 보장이 없다. 같은 길이로
강제하면 **동기화를 이미 했다고 스키마가 조용히 가정**하는 것이고, 어긋난 데이터가
계약 검증기를 통과해 실물에서만 실패한다. 길이를 분리하면 변환기가 리샘플 정책을
명시하지 않고는 돌아가지 않는다 — S15P21A103-30 이 꽂히는 자리다.

**2. 절대 pose 를 저장한다. 델타가 아니다.**
기획서의 액션 표현은 EEF 델타지만, 델타는 프레임 하나가 유실되면 복구가 불가능하고
어느 기준에서의 델타인지 기록에 남지 않는다. 델타로 바꾸는 것은 다운스트림의 선택
사항이다. raw 는 가장 덜 가공된 형태로 남긴다.

**3. 프레임과 캘리브레이션을 필드로 못 박고, 없으면 거부한다.**
`frame` 이 `robot_base` 가 아니면 변환기가 거부한다. hand-eye `T_cam→tcp` 미실측
(LIMITS L30)이 손측정 근사값으로 조용히 통과하는 경로를 없애기 위한 것이다.
**상수 편향은 학습이 지우지 못한다** — 실측상 ±10mm 에서 재생 성공률이 3/4 로
떨어진다 🟢. 빈 칸은 그럴듯한 값으로 채우지 않고 실패로 만든다.

## 번들에서 여기로 올 때 반드시 변환해야 하는 것 — 전부 조용히 틀리는 종류다

| 번들 (SPEC) | 여기 | 안 바꾸면 |
|---|---|---|
| `quaternion_order: "xyzw"` | **`wxyz`** (MuJoCo 규약) | 180도급 회전 오류. 손실은 잘 떨어지고 실물에서만 실패 |
| `pose_is: "T_world_camera"` | **파지점(pinch) pose** | `T_cam→tcp` 상수 편향. 학습이 지우지 못한다 |
| `frame: "arcore_world"`, **Y = 중력반대** | `robot_base`, **Z-up** | 축 교환을 빠뜨리면 중력 방향이 틀린다 |
| 원점 = 세션 시작 (에피소드마다 다름) | 로봇 베이스 고정 | 상대궤적만으로는 어디로 움직일지 알 수 없다 |

⚠️ 마지막 두 줄의 변환(`T_ARCore월드→로봇베이스`)은 **아직 없다.**
   SPEC 「막는 것」 절이 방안 A(작업대 ArUco 보드)를 권한다. 트랙 A 소관이다.

## 그리퍼 개구 — 양 끝이 겹치지 않는다 🟢

    UMI 리그 실측     0 ~ 90 mm      (SPEC, 핑거 50% 축소 · 마커 스케일)
    SO-101 패드 간격  5.3 ~ 79.4 mm  (configs/so101.yaml grasp.gap_curve)

겹치는 구간은 **5.3~79.4mm** 다. 그 밖의 프레임은 `umi.convert.invert_gap_curve`
가 거부한다 — 사람이 로봇이 못 하는 것을 요구한 것이고, 반올림 오차가 아니라 발견이다.

겹치는 구간에서는 **항등으로 쓴다.** `gap_m` 과 패드 간격은 둘 다 "물체 폭만큼
벌린 정도"를 재고, 핑거 50% 축소는 핑거 **크기**를 바꾸지 기준 개구를 바꾸지 않는다.
근거 없는 리타깃 함수를 끼우면 계통 오차를 만드는 쪽이 된다 🟡 — 트랙 A·HW 확인 대상.

## EEF 프레임 규약 — 부호를 틀리면 조용히 편향된다

`eef_quat` 이 기술하는 회전 R 에 대해:

    R[:, 2]  =  손가락이 향하는 방향 (접근축, approach axis)
    R[:, 0]  =  턱이 열리고 닫히는 방향 (jaw axis)
    R[:, 1]  =  나머지 (오른손 좌표계)

⚠️ 시뮬 그리퍼 body 는 손가락이 **body 로컬 -z** 를 향한다 (실측 🟢,
   `sim/mujoco/kinematics.py:approach_axis`). 어댑터는 부호를 뒤집어야 한다.
   이 규약을 여기 적어두지 않으면 부호 오류가 "IK 가 좀 안 맞네" 로만 보인다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from contract.ids import SKILL_IDS

# Bump when the raw schema changes. Independent of CONTRACT_VERSION: raw is not
# the seam between tracks, the contract is. Raw may move faster.
# raw 스키마가 바뀌면 올린다. CONTRACT_VERSION 과 독립이다 — 트랙 간 접점은 계약이고
# raw 는 아니다. raw 는 더 빨리 움직여도 된다.
RAW_VERSION = "umi_intermediate/0.2.0-provisional"
"""이 중간 표현의 버전. 번들 스키마(`umi_raw/0.1.0`)와 **다른 것**이다.
0.2.0 (2026-09-08): `gripper_status` 추가(SPEC 의 D/M/T/X), 번들 출처 필드 추가."""

BUNDLE_SCHEMA_SUPPORTED = ("umi_raw/0.1.0",)
"""canonical 번들 스키마. 다른 값이면 거부한다.

**정정 (2026-09-12, 김준태).** 0912 오전에 여기 `arpose.episode/1` 을 추가했다.
틀렸다. 되돌린다.

당시 논리는 "이 목록은 합의된 이름이 아니라 우리 파서가 읽을 수 있는 것"이었다.
그 논리대로면 앱이 스키마 이름을 낼 때마다 이 목록이 길어지고, 그 순간부터
`raw.py` 를 통과했다는 사실이 "canonical raw 다" 를 더는 뜻하지 않는다.
검사가 통과 조건을 스스로 넓히면 그건 검사가 아니다.

황도경 제안(2026-09-12)이 맞다 — **경계 어댑터에서 정규화한다.**
  앱 원본 `arpose.episode/1`
    -> 경계 어댑터 (`umi/arcore_pilot.py`, 트랙 A `track_a/convert/arpose_delivery.py`)
    -> canonical `umi_raw/0.1.0`
    -> `raw.py` 는 canonical 만 받는다
어댑터는 원본 스키마 이름을 `meta.notes["bundle_schema_origin"]` 에 남긴다.
출처는 보존되고, 통과 조건은 하나로 고정된다.

⚠️ **SPEC 과 앱의 이름 불일치 자체는 해소되지 않았다.** MW·트랙 A 와 정해야 한다:
   앱이 SPEC 이름으로 바꿀 것인가, SPEC 이 앱 이름을 받을 것인가.
   그 결정 전까지 어댑터가 그 간극을 흡수한다 — `raw.py` 가 아니라."""

GRIPPER_STATUS_VALID = ("D", "M")
"""`gap_m` 이 있는 상태값. D=양쪽 직접검출 · M=중심선 대칭 추정 (SPEC, 약 97%).
T=템플릿 보강 · X=실패 는 `gap_m` 이 **결측**이다 — 채우지 않고 학습에서 제외한다."""

UMI_GAP_RANGE_M = (0.0, 0.090)
"""UMI 리그의 물리 개구 실측 (SPEC). SO-101 패드 간격은 0.0053~0.0794m 다."""

FRAME_ROBOT_BASE = "robot_base"
"""The only frame the converter accepts. Everything else must be transformed first.
변환기가 받는 유일한 프레임. 나머지는 먼저 변환해야 한다."""

KNOWN_FRAMES: tuple[str, ...] = (
    FRAME_ROBOT_BASE,
    "arcore_world",   # ARCore 세션 원점. T_world->base 미적용
    "camera",         # 카메라 광학 좌표계
)

# Sources that describe real recordings. These MUST carry a calibration id --
# a real pose in robot_base frame cannot exist without one having been applied.
# 실기록을 기술하는 source. 캘리브레이션 id 가 **필수**다 — 실제 pose 가
# robot_base 프레임에 있다는 것은 캘리브레이션이 적용됐다는 뜻이고, 그게 없으면
# 누군가 프레임 이름만 바꿔 쓴 것이다.
REAL_SOURCES: tuple[str, ...] = ("arcore", "arcore_imu")

QUAT_NORM_TOL = 1e-6


@dataclass
class RawMeta:
    """Everything needed to judge whether a raw recording is convertible.
    raw 기록을 변환할 수 있는지 판단하는 데 필요한 전부."""

    recording_id: str
    skill_id: str
    source: str            # "arcore" | "arcore_imu" | "sim_synth"
    frame: str             # KNOWN_FRAMES 중 하나
    n_steps: int           # pose 스텝 수 T
    pose_rate_hz: float
    cameras: list[str]
    # Which calibration produced `frame`. Empty means none was applied -- legal
    # only for synthetic sources, which have no camera and no hand-eye at all.
    # 어느 캘리브레이션이 이 `frame` 을 만들었는가. 빈 값은 미적용을 뜻하고,
    # 카메라도 hand-eye 도 애초에 없는 합성 source 에만 허용한다.
    calibration_id: str = ""
    raw_version: str = RAW_VERSION
    # 어느 번들에서 왔는가. `episode.json` 의 `schema` 를 그대로 옮긴다.
    # 빈 값은 번들에서 오지 않았다는 뜻이다 (합성·시뮬 FK).
    bundle_schema: str = ""
    # SPEC `summary.txt` 에서 옮긴다. 로더가 이걸 무시하면 안 되는 구간을 학습한다.
    #   frames_dropped: JPEG 인코딩 지연으로 이미지가 없는 pose 행 수
    #   usable_segments: 추적 점프·PAUSED 를 뺀 사용 가능 구간 [[start, end), ...]
    #                    ⚠️ 초기 3~5초는 ARCore 안정화 전이라 폐기 대상이다 (SPEC)
    frames_dropped: int = 0
    usable_segments: list[list[int]] = field(default_factory=list)
    device: str = ""
    recorded_by: str = ""
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawEpisode:
    """One demonstration as recorded. Time-major, absolute poses, SI units.
    기록된 그대로의 시연 하나. 시간축이 첫 축, 절대 pose, SI 단위.

    Image arrays are HWC uint8 at whatever resolution the camera produced --
    resizing to the contract's (3, 224, 224) CHW is the converter's job, not the
    recorder's. A recorder that resizes has thrown information away before
    anyone decided it was safe to.
    이미지는 카메라가 낸 해상도 그대로 HWC uint8 이다. 계약의 (3,224,224) CHW 로
    줄이는 것은 변환기의 일이고 기록기의 일이 아니다. 기록기가 줄이면 그게
    안전한지 아무도 판단하기 전에 정보를 버린 것이다.
    """

    meta: RawMeta
    eef_pos: np.ndarray                      # (T, 3) float64, metres, `frame` 기준
    eef_quat: np.ndarray                     # (T, 4) float64, wxyz, 단위 사원수
    gripper_gap_m: np.ndarray                # (T,) float64 [m]. 결측은 NaN
    gripper_status: np.ndarray               # (T,) '<U1' — SPEC 의 D/M/T/X
    pose_timestamp: np.ndarray               # (T,) float64 seconds
    images: dict[str, np.ndarray]            # cam -> (Ti, H, W, 3) uint8
    image_timestamp: dict[str, np.ndarray]   # cam -> (Ti,) float64 seconds


class RawError(ValueError):
    """A raw recording violates the raw schema.
    raw 기록이 raw 스키마를 위반했다."""


def validate_raw(ep: RawEpisode) -> list[str]:
    """Return every schema violation found. Empty list means the recording is usable.
    발견된 스키마 위반을 전부 반환한다. 빈 리스트면 쓸 수 있다.

    A list rather than a first-failure exception, for the same reason
    `contract.episode.validate` does it: when a recording session goes wrong you
    need to see all of what is wrong at once, not the first thing.
    첫 위반에서 예외를 던지지 않는 이유는 `contract.episode.validate` 와 같다.
    수집이 잘못됐을 때 무엇이 얼마나 잘못됐는지 한 번에 봐야 한다.
    """
    problems: list[str] = []
    m = ep.meta
    t = m.n_steps

    if t <= 0:
        problems.append(f"n_steps must be positive, got {t}")

    for label, arr, shape, dtype in (
        ("eef_pos", ep.eef_pos, (t, 3), np.float64),
        ("eef_quat", ep.eef_quat, (t, 4), np.float64),
        ("gripper_gap_m", ep.gripper_gap_m, (t,), np.float64),
        ("pose_timestamp", ep.pose_timestamp, (t,), np.float64),
    ):
        if arr.dtype != dtype:
            problems.append(f"{label}.dtype must be {np.dtype(dtype)}, got {arr.dtype}")
        if arr.shape != shape:
            problems.append(f"{label}.shape must be {shape}, got {arr.shape}")
        elif label == "gripper_gap_m":
            pass  # 결측(NaN)은 아래에서 gripper_status 와 함께 검사한다
        elif not np.isfinite(arr).all():
            problems.append(f"{label} contains NaN or inf")

    # gap 은 D·M 프레임에만 있다 (SPEC, 약 97%). T·X 는 결측이고 채우지 않는다 —
    # 0 으로 채우면 "완전히 닫혔다"는 뜻이 되어 학습이 그것을 배운다.
    if ep.gripper_status.shape != (t,):
        problems.append(f"gripper_status.shape must be {(t,)}, got {ep.gripper_status.shape}")
    elif ep.gripper_status.dtype.kind not in ("U", "S"):
        problems.append(f"gripper_status.dtype must be a string kind, got {ep.gripper_status.dtype}")
    elif ep.gripper_gap_m.shape == (t,):
        status = np.asarray(ep.gripper_status).astype("<U1")
        bad = sorted(set(status.tolist()) - set("DMTX"))
        if bad:
            problems.append(f"gripper_status 에 알 수 없는 값 {bad} — SPEC 은 D/M/T/X 다")
        have = np.isin(status, list(GRIPPER_STATUS_VALID))
        if not np.isfinite(ep.gripper_gap_m[have]).all():
            problems.append(
                "gripper_gap_m 이 D·M 프레임에서 결측이다 — 그 상태값은 gap 이 있다는 뜻이다"
            )
        if np.isfinite(ep.gripper_gap_m[~have]).any():
            problems.append(
                "gripper_gap_m 이 T·X 프레임에 값이 있다 — 채우지 않고 NaN 으로 둔다. "
                "0 으로 채우면 '완전히 닫혔다'가 되어 학습이 그것을 배운다"
            )

    if ep.eef_quat.shape == (t, 4) and np.isfinite(ep.eef_quat).all():
        norms = np.linalg.norm(ep.eef_quat, axis=1)
        worst = float(np.abs(norms - 1.0).max()) if t else 0.0
        if worst > QUAT_NORM_TOL:
            problems.append(
                f"eef_quat is not unit: worst |q|-1 = {worst:.3e} > {QUAT_NORM_TOL:.0e} "
                "— 정규화되지 않은 사원수는 회전이 아니다"
            )

    if ep.gripper_gap_m.shape == (t,) and np.isfinite(ep.gripper_gap_m).any():
        if float(np.nanmin(ep.gripper_gap_m)) < 0.0:
            problems.append(
                f"gripper_gap_m has a negative gap (min {float(np.nanmin(ep.gripper_gap_m)):.4f}) "
                "— 손가락 간격은 음수가 될 수 없다"
            )

    if ep.pose_timestamp.shape == (t,) and t > 1:
        if not np.all(np.diff(ep.pose_timestamp) > 0):
            problems.append("pose_timestamp is not strictly increasing")

    if set(ep.images) != set(m.cameras):
        problems.append(
            f"camera mismatch: arrays={sorted(ep.images)} meta={sorted(m.cameras)}"
        )
    if set(ep.image_timestamp) != set(ep.images):
        problems.append(
            f"image_timestamp cameras {sorted(ep.image_timestamp)} "
            f"!= image cameras {sorted(ep.images)}"
        )

    for cam, arr in ep.images.items():
        if arr.dtype != np.uint8:
            problems.append(f"images[{cam}].dtype must be uint8, got {arr.dtype}")
        if arr.ndim != 4 or arr.shape[-1] != 3:
            problems.append(
                f"images[{cam}].shape must be (Ti, H, W, 3) HWC, got {arr.shape}"
            )
            continue
        ts = ep.image_timestamp.get(cam)
        if ts is None:
            continue
        if ts.dtype != np.float64:
            problems.append(f"image_timestamp[{cam}].dtype must be float64, got {ts.dtype}")
        if ts.shape != (arr.shape[0],):
            problems.append(
                f"image_timestamp[{cam}].shape must be {(arr.shape[0],)} "
                f"to match {arr.shape[0]} frames, got {ts.shape}"
            )
        elif arr.shape[0] > 1 and not np.all(np.diff(ts) > 0):
            problems.append(f"image_timestamp[{cam}] is not strictly increasing")

    if m.frame not in KNOWN_FRAMES:
        problems.append(f"frame {m.frame!r} 이 {KNOWN_FRAMES} 에 없다")

    if m.source in REAL_SOURCES and not m.calibration_id:
        problems.append(
            f"source {m.source!r} 인데 calibration_id 가 비어 있다 "
            "— 실기록이 robot_base 프레임에 있다는 것은 캘리브레이션이 적용됐다는 뜻이다. "
            "id 가 없으면 프레임 이름만 바꿔 쓴 것이고, 그 오차는 상수 편향이라 "
            "학습이 지우지 못한다 (LIMITS L30)"
        )

    if m.skill_id not in SKILL_IDS:
        problems.append(f"skill_id {m.skill_id!r} 이 {SKILL_IDS} 에 없다")

    if m.raw_version != RAW_VERSION:
        problems.append(f"raw_version {m.raw_version!r} != {RAW_VERSION!r}")

    if m.source in REAL_SOURCES and m.bundle_schema not in BUNDLE_SCHEMA_SUPPORTED:
        problems.append(
            f"source {m.source!r} 인데 bundle_schema 가 {m.bundle_schema!r} 다. "
            f"지원 {BUNDLE_SCHEMA_SUPPORTED} — 번들 스키마가 바뀌면 파서도 바뀌어야 한다"
        )

    if m.pose_rate_hz <= 0:
        problems.append(f"pose_rate_hz must be positive, got {m.pose_rate_hz}")

    return problems


def require_convertible(ep: RawEpisode) -> None:
    """Raise unless this recording may be converted into a contract episode.
    이 기록을 계약 에피소드로 변환해도 되는 상태가 아니면 예외를 던진다.

    Stricter than `validate_raw`: the frame must already be the robot's base.
    Transforming into it is track A's job (S15P21A103-29), and doing it silently
    here would hide an unmeasured constant offset inside the converter.
    `validate_raw` 보다 엄격하다. 프레임이 이미 로봇 베이스여야 한다. 거기로
    변환하는 것은 트랙 A 의 일(S15P21A103-29)이고, 여기서 조용히 하면 미계측
    상수 오프셋을 변환기 안에 숨기는 것이 된다.
    """
    problems = validate_raw(ep)
    if ep.meta.frame != FRAME_ROBOT_BASE:
        problems.append(
            f"frame 이 {ep.meta.frame!r} 다. 변환기는 {FRAME_ROBOT_BASE!r} 만 받는다 "
            "— 프레임 변환은 S15P21A103-29 (트랙 A) 소관이다. 여기서 하지 않는다"
        )
    if problems:
        raise RawError(
            "raw 스키마 위반 상태로는 변환하지 않는다:\n  " + "\n  ".join(problems)
        )


def write_raw(ep: RawEpisode, out_dir: Path) -> Path:
    """Write one raw recording as an .npz plus a sidecar .json of its metadata.
    raw 기록 하나를 .npz 와 메타데이터 .json 으로 쓴다.

    Unlike `contract.episode.write_episode` this refuses only on *schema*
    violations, not on convertibility. A recording taken in the camera frame is
    a legitimate thing to have on disk -- it is just not yet convertible.
    `contract.episode.write_episode` 와 달리 **스키마** 위반만 거부하고 변환
    가능성은 따지지 않는다. 카메라 프레임에서 찍힌 기록은 디스크에 있어도 되는
    정당한 물건이다. 아직 변환할 수 없을 뿐이다.
    """
    problems = validate_raw(ep)
    if problems:
        raise RawError("raw 스키마 위반 상태로는 저장하지 않는다:\n  " + "\n  ".join(problems))

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{ep.meta.recording_id}.raw.npz"
    arrays: dict[str, np.ndarray] = {
        "eef_pos": ep.eef_pos,
        "eef_quat": ep.eef_quat,
        "gripper_gap_m": ep.gripper_gap_m,
        "gripper_status": np.asarray(ep.gripper_status).astype("<U1"),
        "pose_timestamp": ep.pose_timestamp,
    }
    for cam, arr in ep.images.items():
        arrays[f"image__{cam}"] = arr
        arrays[f"image_timestamp__{cam}"] = ep.image_timestamp[cam]
    np.savez_compressed(npz_path, **arrays)
    (out_dir / f"{ep.meta.recording_id}.raw.json").write_text(
        json.dumps(asdict(ep.meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return npz_path


def read_raw(npz_path: Path) -> RawEpisode:
    """Read back a recording written by :func:`write_raw`.
    write_raw 로 쓴 기록을 다시 읽는다."""
    meta_path = npz_path.with_suffix("").with_suffix(".raw.json")
    if not meta_path.exists():
        raise RawError(f"메타데이터가 없다: {meta_path}")
    meta = RawMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
    with np.load(npz_path) as z:
        images = {
            k[len("image__"):]: z[k]
            for k in z.files
            if k.startswith("image__")
        }
        stamps = {
            k[len("image_timestamp__"):]: z[k]
            for k in z.files
            if k.startswith("image_timestamp__")
        }
        return RawEpisode(
            meta=meta,
            eef_pos=z["eef_pos"],
            eef_quat=z["eef_quat"],
            gripper_gap_m=z["gripper_gap_m"],
            gripper_status=z["gripper_status"],
            pose_timestamp=z["pose_timestamp"],
            images=images,
            image_timestamp=stamps,
        )
