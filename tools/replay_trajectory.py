"""Replay a sim joint trajectory on the real SO-101 and log tracking error.
시뮬 관절궤적을 실물 SO-101 에 재생하고 추종 오차를 기록한다.

왜 필요한가 (2026-09-18)
------------------------
모터 교체로 실물이 살아났다. 지금까지 모든 수치가 시뮬 안에서만 나왔고
**sim2real 갭이 0건 측정**이다.

갭을 한꺼번에 재면 실패해도 원인을 모른다. 그래서 가장 아래 칸부터 연다.

    1a  물체 없이 궤적 재생 → 명령 vs 실제 관절각   = **순수 제어 갭**
    1b  물체 놓고 재생 → 집히나                    = 물리 갭
    2   같은 장면의 실물/시뮬 이미지로 액션 비교      = 지각 갭
    3   폐루프 롤아웃                              = 전체

**이 도구는 1a·1b 를 담당한다.** 정책도 카메라도 IK 도 안 쓴다.
팔이 명령한 자리로 가는지만 본다. 여기서 무너지면 위 칸은 볼 필요도 없다.

환산식 — 추측하지 않고 소스에서 가져왔다 🟢
--------------------------------------------
`so101_system.cpp` / `so101_system.hpp` 원문:

    TICKS_PER_TURN = 4096.0
    tick = zero_tick + direction * rad * TICKS_PER_TURN / (2*PI)
    rad  = (tick - zero_tick) * direction * (2*PI) / TICKS_PER_TURN
    gripper_tick = closed_tick + (open_tick - closed_tick) * width_m / max_width_m

⚠️ **0.09 도/틱이 아니라 0.087890625 도/틱이다.** HW 가 구두로 준 0.09 는 반올림이다.
   738틱을 0.09 로 곱하면 66.42도, 소스대로면 64.86도다. 블렌더 간섭검사의 +65도와
   64.86 이 맞는다. **남이 준 파생값 말고 원시 틱과 소스 환산식을 쓴다.**

안전 — ros2_control 이 해주던 걸 우리가 해야 한다
--------------------------------------------------
직접 경로는 `SO101System` 의 검사를 우회한다. 그래서 같은 검사를 여기서 한다.

    - rad 이 [min_rad, max_rad] 밖이면 거부 (so101_system.cpp 와 동일)
    - 환산된 tick 이 [min_tick, max_tick] 밖이면 거부 (동일)
    - **브래킷 한계** wrist_roll <= +64.86도 (소스에 없다. 우리가 잰 것)
    - 한 번에 움직이는 각도를 --max-step-deg 로 제한하고 그 사이를 보간한다
    - 전체 궤적을 **한 스텝도 빠짐없이 먼저 검사**하고, 하나라도 걸리면 아예 안 움직인다

⚠️ 이 도구는 팔을 실제로 움직인다. `--dry-run` 으로 먼저 보고, 팔 주변을 비우고,
   비상시 전원을 끊을 수 있는 상태에서 돌린다.

Usage
-----
  python3 replay_trajectory.py --selftest
  python3 replay_trajectory.py --port /dev/ttyTHS1 --read-only
  python3 replay_trajectory.py --port /dev/ttyTHS1 --pose "0,-0.77,1.09,-0.32,0" --dry-run
  python3 replay_trajectory.py --port /dev/ttyTHS1 --pose "0,-0.77,1.09,-0.32,0" --yes
  python3 replay_trajectory.py --port /dev/ttyTHS1 --trajectory traj.json --out track.json --yes
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

TICKS_PER_TURN = 4096.0
PING, READ, WRITE = 0x01, 0x02, 0x03
TORQUE_ENABLE, PRESENT_POSITION = 40, 56

# so101.ros2_control.xacro 원문 값. direction 은 BACKEND_INTERFACE "Verified" 표에서 전부 +1
JOINTS = [
    # name,            id, zero, dir, min_tick, max_tick, min_rad,        max_rad
    ("shoulder_pan",    1, 1990, +1, 0, 4095, -1.91986,       +1.91986),
    ("shoulder_lift",   2, 2048, +1, 0, 4095, -1.74533,       +1.74533),
    ("elbow_flex",      3, 1024, +1, 0, 4095, -1.57079632679, +1.69),
    ("wrist_flex",      4, 2048, +1, 0, 4095, -1.65806,       +1.65806),
    ("wrist_roll",      5, 2126, +1, 0, 4095, -2.743847297,   +2.841206309),
]
GRIPPER = ("gripper", 6, 1720, 70, 70, 1720, 0.09)      # name,id,closed,open,min,max,max_width

# 소스에 없는 물리 제약. 우리가 측정한 것
BRACKET_WRIST_ROLL_MAX_RAD = math.radians(738 * 360.0 / TICKS_PER_TURN)   # 64.86도
BRACKET_NOTE = ("폰 홀더 L 브래킷. HW 실측 2864틱 · 블렌더 간섭검사 +65도. "
                "브래킷 수정되면 갱신한다")


# ── 환산 — so101_system.cpp 와 바이트 단위로 같은 식 ──────────────────────

def joint_to_tick(j: tuple, radians: float) -> int:
    """rad -> servo tick, refusing anything the ROS plugin would refuse.
    rad → 틱. ROS 플러그인이 거부하는 것은 여기서도 거부한다."""
    name, _id, zero, dirn, mn_t, mx_t, mn_r, mx_r = j
    if not math.isfinite(radians) or radians < mn_r or radians > mx_r:
        raise ValueError(f"{name}: {math.degrees(radians):+.2f}도 가 관절 한계 "
                         f"[{math.degrees(mn_r):+.2f}, {math.degrees(mx_r):+.2f}] 밖이다")
    raw = zero + dirn * radians * TICKS_PER_TURN / (2.0 * math.pi)
    if raw < mn_t or raw > mx_t:
        raise ValueError(f"{name}: 틱 {raw:.1f} 이 [{mn_t}, {mx_t}] 밖이다")
    return int(round(raw))


def tick_to_joint(j: tuple, tick: int) -> float:
    """servo tick -> rad. 틱 → rad."""
    _n, _i, zero, dirn, *_ = j
    if not 0 <= tick <= 4095:
        raise ValueError(f"틱 {tick} 이 0..4095 밖이다")
    return (tick - zero) * dirn * (2.0 * math.pi) / TICKS_PER_TURN


def gripper_to_tick(width_m: float) -> int:
    """gripper full opening [m] -> tick. 전체 개구[m] → 틱."""
    _n, _i, closed, opened, mn, mx, mw = GRIPPER
    if not math.isfinite(width_m) or width_m < 0.0 or width_m > mw:
        raise ValueError(f"gripper: 폭 {width_m} 이 0..{mw} 밖이다")
    raw = closed + (opened - closed) * width_m / mw
    if raw < mn or raw > mx:
        raise ValueError(f"gripper: 틱 {raw:.1f} 이 [{mn}, {mx}] 밖이다")
    return int(round(raw))


def tick_to_gripper(tick: int) -> float:
    _n, _i, closed, opened, *_rest, mw = GRIPPER
    return (tick - closed) * mw / (opened - closed)


def check_extra_limits(q: list[float]) -> list[str]:
    """Physical limits the ROS plugin does not know about. 플러그인이 모르는 물리 제약."""
    bad = []
    if len(q) >= 5 and q[4] > BRACKET_WRIST_ROLL_MAX_RAD:
        bad.append(f"wrist_roll {math.degrees(q[4]):+.2f}도 > 브래킷 한계 "
                   f"{math.degrees(BRACKET_WRIST_ROLL_MAX_RAD):+.2f}도 ({BRACKET_NOTE})")
    return bad


def interpolate(a: list[float], b: list[float], max_step_rad: float) -> list[list[float]]:
    """Split one jump into steps no joint exceeds max_step_rad. 한 도약을 잘게 쪼갠다."""
    d = max(abs(y - x) for x, y in zip(a, b))
    n = max(1, int(math.ceil(d / max_step_rad)))
    return [[x + (y - x) * (k + 1) / n for x, y in zip(a, b)] for k in range(n)]


# ── 시리얼 ───────────────────────────────────────────────────────────────

def checksum(body: list[int]) -> int:
    return (~sum(body)) & 0xFF


def packet(sid: int, inst: int, params: list[int] | None = None) -> bytes:
    p = params or []
    body = [sid, len(p) + 2, inst, *p]
    return bytes([0xFF, 0xFF, *body, checksum(body)])


class Bus:
    """Minimal ST3215 bus with position read/write. 위치 읽기·쓰기 최소 버스."""

    def __init__(self, port: str, baud: int = 1000000, timeout: float = 0.2) -> None:
        import serial
        self.s = serial.Serial(port, baud, timeout=timeout)

    def close(self) -> None:
        self.s.close()

    def request(self, sid: int, inst: int, params: list[int] | None = None):
        self.s.reset_input_buffer()
        self.s.write(packet(sid, inst, params)); self.s.flush()
        head = self.s.read(4)
        if len(head) != 4 or head[0] != 0xFF or head[1] != 0xFF:
            return None
        rid, length = head[2], head[3]
        rest = self.s.read(length)
        if len(rest) != length or rid != sid:
            return None
        if checksum([rid, length, *rest[:-1]]) != rest[-1]:
            return None
        return list(rest[1:-1])

    def position(self, sid: int):
        d = self.request(sid, READ, [PRESENT_POSITION, 2])
        return None if d is None or len(d) != 2 else d[0] | (d[1] << 8)

    def torque(self, sid: int, on: bool) -> bool:
        return self.request(sid, WRITE, [TORQUE_ENABLE, 1 if on else 0]) is not None

    def move(self, sid: int, tick: int, speed: int = 300, accel: int = 10) -> bool:
        p = [42, accel,
             tick & 0xFF, tick >> 8, 0, 0, speed & 0xFF, speed >> 8]
        return self.request(sid, WRITE, p) is not None


# ── 자체 검증 ────────────────────────────────────────────────────────────

def selftest() -> int:
    bad = 0
    wr = JOINTS[4]

    ok = joint_to_tick(wr, 0.0) == 2126
    print(f"[1] wrist_roll 0 rad → 틱 {joint_to_tick(wr, 0.0)}  기대 2126  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    deg = math.degrees(tick_to_joint(wr, 2864))
    ok = abs(deg - 64.86328125) < 1e-6
    print(f"[2] 틱 2864 → {deg:.6f}도  기대 64.863281  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    r = math.radians(30.0)
    back = tick_to_joint(wr, joint_to_tick(wr, r))
    ok = abs(math.degrees(back) - 30.0) < 0.088       # 양자화 1틱
    print(f"[3] 30도 왕복 → {math.degrees(back):.4f}도 (양자화 0.0879도)  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    try:
        joint_to_tick(wr, math.radians(200.0)); caught = False
    except ValueError:
        caught = True
    print(f"[4] 200도 거부  ", end=""); print("OK" if caught else "!! 실패 — 통과시켰다")
    bad += (not caught)

    g = [(0.0, 1720), (0.09, 70), (0.045, 895)]
    ok = all(gripper_to_tick(w) == t for w, t in g)
    print(f"[5] 그리퍼 0/0.045/0.09 m → {[gripper_to_tick(w) for w,_ in g]}  "
          f"기대 [1720, 895, 70]  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    over = [0, 0, 0, 0, math.radians(70.0)]
    ok = len(check_extra_limits(over)) == 1 and not check_extra_limits([0, 0, 0, 0, 0])
    print(f"[6] 브래킷 70도 감지 / 0도 통과  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    steps = interpolate([0.0] * 5, [math.radians(30.0)] + [0.0] * 4, math.radians(3.0))
    ok = len(steps) == 10 and abs(math.degrees(steps[-1][0]) - 30.0) < 1e-9
    print(f"[7] 30도를 3도씩 → {len(steps)}스텝, 끝 {math.degrees(steps[-1][0]):.4f}도  "
          f"기대 10스텝 30도  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    ok = packet(1, PING) == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])
    print(f"[8] PING 프레임 바이트 대조  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


# ── 본체 ─────────────────────────────────────────────────────────────────

def read_all(bus: Bus) -> tuple[list[float] | None, float | None, int]:
    q, okn = [], 0
    for j in JOINTS:
        t = bus.position(j[1])
        if t is None:
            q.append(float("nan"))
        else:
            q.append(tick_to_joint(j, t)); okn += 1
    gt = bus.position(GRIPPER[1])
    return q, (tick_to_gripper(gt) if gt is not None else None), okn


def preflight(waypoints: list[list[float]]) -> None:
    """Check EVERY step before moving anything. 한 스텝도 빠짐없이 먼저 검사한다."""
    errs = []
    for i, q in enumerate(waypoints):
        for j, r in zip(JOINTS, q):
            try:
                joint_to_tick(j, r)
            except ValueError as e:
                errs.append(f"  스텝 {i}: {e}")
        for m in check_extra_limits(q):
            errs.append(f"  스텝 {i}: {m}")
    print(f"사전검사 {len(waypoints)}스텝 · 위반 {len(errs)}건")
    if errs:
        for e in errs[:20]:
            print(e)
        if len(errs) > 20:
            print(f"  ... 외 {len(errs)-20}건")
        raise SystemExit("!! 위반이 있어 한 스텝도 움직이지 않는다")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--port", default="/dev/ttyTHS1")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--read-only", action="store_true", help="현재 자세만 읽고 끝")
    ap.add_argument("--pose", help="목표 관절각 5개, rad, 콤마 구분")
    ap.add_argument("--trajectory", help="[[q1..q5], ...] 형태 JSON 파일")
    ap.add_argument("--gripper", type=float, default=None, help="개구 m (0~0.09)")
    ap.add_argument("--max-step-deg", type=float, default=3.0)
    ap.add_argument("--period", type=float, default=0.05, help="명령 간격 초")
    ap.add_argument("--speed", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="확인 프롬프트 생략")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 팔을 건드리지 않는다")
    print()

    try:
        bus = Bus(a.port, a.baud)
    except Exception as exc:                                   # noqa: BLE001
        raise SystemExit(f"!! 포트를 못 연다: {type(exc).__name__}: {exc}")

    if bus.position(200) is not None:
        bus.close()
        raise SystemExit("!! 없는 ID 200 이 응답했다. 버스가 이상하다. 중단")
    print("[검정] 없는 ID 200 무응답 OK")

    q0, g0, okn = read_all(bus)
    print(f"현재 자세 읽기 {okn}/{len(JOINTS)}")
    for j, r in zip(JOINTS, q0):
        print(f"  {j[0]:14s} {math.degrees(r):+8.2f}도")
    print(f"  {'gripper':14s} {'읽기실패' if g0 is None else f'{g0*1000:8.2f} mm'}")
    if a.read_only or okn < len(JOINTS):
        if okn < len(JOINTS):
            print("!! 일부 관절을 못 읽는다. 움직이지 않는다")
        bus.close(); return

    if a.pose:
        target = [float(x) for x in a.pose.split(",")]
        if len(target) != 5:
            raise SystemExit("!! --pose 는 관절각 5개다")
        waypoints = [target]
    elif a.trajectory:
        waypoints = json.loads(Path(a.trajectory).read_text(encoding="utf-8"))
    else:
        bus.close(); raise SystemExit("!! --pose 또는 --trajectory 가 필요하다")

    steps: list[list[float]] = []
    cur = q0
    for w in waypoints:
        seg = interpolate(cur, w, math.radians(a.max_step_deg))
        steps.extend(seg); cur = w
    preflight(steps)

    dur = len(steps) * a.period
    print(f"\n계획: 경유점 {len(waypoints)} → 보간 {len(steps)}스텝 · "
          f"스텝당 최대 {a.max_step_deg}도 · 약 {dur:.1f}초")
    if a.dry_run:
        print("--dry-run 이므로 움직이지 않는다"); bus.close(); return
    if not a.yes:
        if input("팔 주변을 비웠나? 진행하려면 'go' 입력: ").strip() != "go":
            print("중단"); bus.close(); return

    log = []
    try:
        for i, q in enumerate(steps):
            ticks = [joint_to_tick(j, r) for j, r in zip(JOINTS, q)]
            for j, t in zip(JOINTS, ticks):
                bus.move(j[1], t, speed=a.speed)
            if a.gripper is not None:
                bus.move(GRIPPER[1], gripper_to_tick(a.gripper), speed=a.speed)
            time.sleep(a.period)
            qa, ga, _ = read_all(bus)
            err = [math.degrees(c - m) for c, m in zip(q, qa)]
            log.append({"step": i, "cmd_rad": q, "act_rad": qa, "err_deg": err,
                        "gripper_m": ga})
            if i % 10 == 0 or i == len(steps) - 1:
                print(f"  {i+1}/{len(steps)}  최대오차 "
                      f"{max(abs(e) for e in err if math.isfinite(e)):.2f}도", end="\r")
    except KeyboardInterrupt:
        print("\n!! 중단됨. 마지막 명령 자세에서 멈춘다")
    finally:
        bus.close()

    print()
    if log:
        import statistics as st
        print(f"\n추종 오차 (명령 − 실제) · {len(log)}스텝")
        for k, j in enumerate(JOINTS):
            e = [abs(r["err_deg"][k]) for r in log if math.isfinite(r["err_deg"][k])]
            if e:
                print(f"  {j[0]:14s} 중앙 {st.median(e):5.2f}도  최대 {max(e):5.2f}도  "
                      f"n {len(e)}/{len(log)}")
        if a.out:
            Path(a.out).write_text(json.dumps(log, indent=1), encoding="utf-8")
            print(f"→ {a.out}")


if __name__ == "__main__":
    main()
