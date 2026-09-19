"""The scripted half of every skill — what happens after the object is held.
스킬의 스크립트 절반. 물체를 쥔 뒤의 전부.

D-AI-75 (놓기 스크립트는 트랙 B 가 짠다) · D-AI-22 (스킬 = 집기 정책 + 놓기 스크립트)

왜 이 파일이 필요한가
---------------------
스킬 5개의 **차이가 전부 여기서 난다.** 집기는 정책 하나를 공유하므로, 놓기가
없으면 `shared_policy_report()` 가 말하는 그대로 **"실제로 동작하는 스킬 0개"** 다.

무엇을 내나
-----------
`policy_to_joints.py` 와 **같은 계약**으로 낸다 — 관절 5 + gap_m 6열 · 점 간격 0.1초.
그래야 집기와 놓기를 한 줄로 이어 붙일 수 있다.

좌표는 우리 것이 아니다
-----------------------
트레이·지그 위치는 `AI/configs/real/place_destinations.yaml` 에서 읽는다.
그 파일이 `status: measured` 가 아니면 **궤적을 내지 않는다** — 예시 좌표로
실물을 움직이면 사고다. dry-run 은 `--allow-provisional` 로 연다.

Usage
-----
  ~/envs/handoff312/bin/python AI/tools/place_scripts.py --selftest
  # [서버] 드라이런 (예시 좌표 허용)
  cd ~/handoff && ~/envs/handoff312/bin/python ~/S15P21A103/AI/tools/place_scripts.py \
      --task configs/can_side.yaml --skill sort_two --branch left_tray \
      --allow-provisional --out-replay outputs/place_sort_two.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract.ids import DESTINATIONS, SKILL_IDS  # noqa: E402

DT_S = 0.1
CFG_DEFAULT = "AI/configs/real/place_destinations.yaml"

# 스킬 -> 놓기 목적지. registry/skills/*.json 의 post_actions 와 같아야 한다.
SKILL_PLACE: dict[str, tuple[str, ...]] = {
    "pick_place": ("target_pose",),
    "sort_two": ("left_tray", "right_tray"),
    "align_fixture": ("fixture",),
    "present_inspect": ("origin",),
    "line_up": ("target_pose",),
}


def rz(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return T


def pose_at(xyz, yaw_deg: float, R_hold: np.ndarray) -> np.ndarray:
    """Keep the grasp orientation, yawed about world z. 파지 자세를 유지하고 z 로만 돌린다.

    자세를 새로 만들지 않는다 — 쥐고 있는 물체의 자세가 바뀌면 놓을 때 쓰러진다."""
    T = np.eye(4)
    T[:3, :3] = rz(yaw_deg)[:3, :3] @ np.asarray(R_hold)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def plan(skill: str, cfg: dict, T_hold: np.ndarray, branch: str | None = None,
         line_index: int = 0) -> list[dict]:
    """TCP poses + gripper opening for the place phase. 놓기 구간의 TCP pose 와 개구.

    한 점 = {T, gap_m, why}. 시간은 점 간격 0.1초로 뒤에서 붙인다.
    **개방은 마지막 직전 한 번뿐이다** — 중간에 열리면 물체를 떨어뜨린다.
    """
    if skill not in SKILL_PLACE:
        raise SystemExit(f"!! 모르는 스킬 {skill!r}. {sorted(SKILL_PLACE)} 중 하나")
    allowed = SKILL_PLACE[skill]
    dest = branch or allowed[0]
    if dest not in allowed:
        raise SystemExit(f"!! {skill} 의 목적지는 {allowed} 다. {dest!r} 는 아니다")
    if dest not in DESTINATIONS:
        raise SystemExit(f"!! {dest!r} 가 계약 DESTINATIONS 에 없다")

    d = cfg["destinations"][dest]
    clear = float(cfg["approach_clearance_m"])
    retreat = float(cfg["retreat_m"])
    hold = float(cfg["hold_gap_m"])
    opened = float(cfg["open_gap_m"])
    R_hold = T_hold[:3, :3]

    xyz = list(map(float, d["xyz"]))
    yaw = float(d.get("yaw_deg", 0.0))

    if skill == "line_up":
        step = float(d["line_up_step_m"])
        count = int(d["line_up_count"])
        if not 0 <= line_index < count:
            raise SystemExit(f"!! line_index {line_index} 가 0~{count-1} 밖이다")
        xyz[1] += step * line_index

    pts: list[dict] = []

    def add(T, gap, why):
        pts.append({"T": T, "gap_m": float(gap), "why": why})

    # 1) 쥔 자리에서 수직으로 들어올린다 (테이블·다른 물체 회피)
    up = T_hold.copy()
    up[2, 3] += clear
    add(T_hold.copy(), hold, "현재 파지 자세")
    add(up, hold, f"수직 상승 {clear*1000:.0f}mm")

    if skill == "present_inspect":
        p = cfg["present"]
        show = pose_at(p["xyz"], yaw, R_hold)
        add(show, hold, "제시 자세로 이동")
        add(pose_at(p["xyz"], yaw + float(p["rotate_deg"]), R_hold), hold, "좌로 회전")
        add(pose_at(p["xyz"], yaw - float(p["rotate_deg"]), R_hold), hold, "우로 회전")
        add(show, hold, "정면 복귀")

    # 2) 목적지 위로 수평 이동 → 하강
    above = pose_at([xyz[0], xyz[1], xyz[2] + clear], yaw, R_hold)
    down = pose_at(xyz, yaw, R_hold)
    add(above, hold, f"{dest} 위로 이동")
    add(down, hold, f"{dest} 로 하강")

    # 3) 개방 — 여기 한 번뿐이다
    add(down.copy(), opened, "개방")

    # 4) 후퇴
    back = down.copy()
    back[2, 3] += retreat
    add(back, opened, f"수직 후퇴 {retreat*1000:.0f}mm")
    return pts


def load_cfg(path: Path, allow_provisional: bool) -> dict:
    if not path.exists():
        raise SystemExit(f"!! 설정이 없다: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    st = cfg.get("status")
    print(f"[설정] {path}  status={st!r}")
    if st != "measured" and not allow_provisional:
        raise SystemExit(
            f"!! status 가 {st!r} 다. 좌표가 실측이 아니므로 궤적을 내지 않는다.\n"
            "   트레이·지그 실제 위치는 김현석, 로봇 기준 좌표는 황도경이 채운다.\n"
            "   드라이런만 하려면 --allow-provisional 을 붙여라 (실물 금지)")
    if st != "measured":
        print("   ⚠️ 예시 좌표다. **실물에 올리지 마라.**")
    return cfg


def selftest() -> int:
    ok = tot = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok, tot
        tot += 1
        ok += bool(cond)
        print(f"  {'OK ' if cond else '!! '} [{tot}] {label:<50} {detail}")

    check(f"스킬 {len(SKILL_PLACE)} / 계약 {len(SKILL_IDS)} 개 전부 정의",
          set(SKILL_PLACE) == set(SKILL_IDS))
    check("목적지가 전부 계약 DESTINATIONS 안",
          all(d in DESTINATIONS for v in SKILL_PLACE.values() for d in v))

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/real/place_destinations.yaml")
        .read_text(encoding="utf-8"))
    T = np.eye(4)
    T[:3, 3] = [0.40, 0.0, 0.03]

    for sk in SKILL_IDS:
        pts = plan(sk, cfg, T)
        opens = [i for i, p in enumerate(pts) if p["gap_m"] > cfg["hold_gap_m"] + 1e-9]
        tot += 1
        good = len(pts) >= 5 and len(opens) == 2 and opens == [len(pts) - 2, len(pts) - 1]
        ok += good
        print(f"  {'OK ' if good else '!! '} [{tot}] {sk:<16} 점 {len(pts)}개 · "
              f"개방 시점 {opens} (마지막 둘이어야 한다)")

    # 판별력: 중간에 개방되는 계획은 잡혀야 한다
    bad = plan("pick_place", cfg, T)
    bad[1]["gap_m"] = cfg["open_gap_m"]
    opens = [i for i, p in enumerate(bad) if p["gap_m"] > cfg["hold_gap_m"] + 1e-9]
    check("판별력: 중간 개방이 섞이면 위치가 어긋난다",
          opens != [len(bad) - 2, len(bad) - 1], str(opens))

    p0 = plan("sort_two", cfg, T, branch="left_tray")
    p1 = plan("sort_two", cfg, T, branch="right_tray")
    check("sort_two 두 분기의 목적지가 다르다",
          not np.allclose(p0[-1]["T"][:3, 3], p1[-1]["T"][:3, 3]))
    try:
        plan("sort_two", cfg, T, branch="fixture")
        died = False
    except SystemExit:
        died = True
    check("판별력: 허용 안 된 분기는 거부", died)

    a = plan("line_up", cfg, T, line_index=0)[-1]["T"][1, 3]
    b = plan("line_up", cfg, T, line_index=2)[-1]["T"][1, 3]
    step = float(cfg["destinations"]["target_pose"]["line_up_step_m"])
    check("line_up 인덱스 2칸이 step*2 만큼 떨어진다",
          abs((b - a) - 2 * step) < 1e-12, f"{(b-a)*1000:.1f}mm")
    try:
        plan("line_up", cfg, T, line_index=99)
        died2 = False
    except SystemExit:
        died2 = True
    check("판별력: 범위 밖 인덱스는 거부", died2)

    pi = plan("present_inspect", cfg, T)
    yaws = [float(np.degrees(np.arctan2(p["T"][1, 0], p["T"][0, 0]))) for p in pi]
    check("present_inspect 이 좌우로 흔든다 (yaw 가 양·음 둘 다)",
          max(yaws) > 1.0 and min(yaws) < -1.0, f"{min(yaws):.1f}~{max(yaws):.1f}도")

    check("파지 자세를 새로 만들지 않는다 (z 회전만)",
          np.allclose(plan("pick_place", cfg, T)[0]["T"][:3, :3], T[:3, :3]))

    print(f"자체검증 {ok} / {tot}")
    return 0 if ok == tot else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--task", help="handoff task yaml")
    ap.add_argument("--skill", choices=sorted(SKILL_PLACE))
    ap.add_argument("--branch", help="sort_two 전용: left_tray / right_tray")
    ap.add_argument("--line-index", type=int, default=0, help="line_up 전용 칸 번호")
    ap.add_argument("--config", default=None, help=f"기본 {CFG_DEFAULT}")
    ap.add_argument("--allow-provisional", action="store_true",
                    help="예시 좌표로 드라이런만. **실물 금지**")
    ap.add_argument("--from-q", help="현재 관절각 5개 rad, 콤마 구분")
    ap.add_argument("--ik-tol", type=float, default=0.008)
    ap.add_argument("--jaw-tol-deg", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-replay", default=None)
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not (a.task and a.skill):
        ap.error("--task 와 --skill 이 필요하다 (또는 --selftest)")

    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 궤적을 내지 않는다")
    print()

    cfg_path = Path(a.config).expanduser() if a.config else \
        Path(__file__).resolve().parents[2] / CFG_DEFAULT
    cfg = load_cfg(cfg_path, a.allow_provisional)

    import mujoco
    import policy_to_joints as p2j
    from simulation.env import PickEnv

    env = p2j.make_env(PickEnv, a.task)
    p2j.tcp_gate(p2j.probe_tcp_offset(env), None)
    jaw = p2j.jaw_gate(p2j.probe_jaw_axis(env), 0.0)

    if a.from_q:
        q0 = np.array([float(v) for v in a.from_q.split(",")], dtype=float)
    else:
        q0 = np.asarray(env.home_q, dtype=float)
        print("[시작] --from-q 미지정 → 태스크 홈 자세에서 시작한다")
    env.ik_data.qpos[env.qids] = q0
    mujoco.mj_forward(env.model, env.ik_data)
    T_hold = env.tcp(env.ik_data).copy()
    print(f"[쥔 자세] {np.round(T_hold[:3, 3], 4)}")

    pts = plan(a.skill, cfg, T_hold, a.branch, a.line_index)
    sol = p2j.solve_waypoints(env, [p["T"] for p in pts], q0, jaw, a.ik_tol, a.jaw_tol_deg)
    good = [s for s in sol if s["q"] is not None and s["reject_reason"] is None]

    print(f"\n스킬 {a.skill} · 목적지 {a.branch or SKILL_PLACE[a.skill][0]}")
    print(f"IK 성공 {len(good)} / 요청 {len(sol)}")
    for s, p in zip(sol, pts):
        mark = "OK" if s["reject_reason"] is None and s["q"] is not None else "거부"
        print(f"  [{s['index']}] {mark:<4} gap {p['gap_m']*1000:5.1f}mm  "
              f"잔차 {s['position_residual_m']*1000:7.3f}mm  {p['why']}"
              f"{'  ' + (s['reject_reason'] or '')}")
    print(f"소요 {len(pts) * DT_S:.1f} s (점 간격 {DT_S}s)")

    report = {
        "_WHAT": "놓기 구간 궤적. 집기(정책) 다음에 이어 붙인다",
        "_contract": "관절5 + gap_m 6열 · 점 간격 0.1s — policy_to_joints 와 동일",
        "_config_status": cfg.get("status"),
        "skill": a.skill, "destination": a.branch or SKILL_PLACE[a.skill][0],
        "line_index": a.line_index, "jaw_offset_deg": jaw,
        "waypoints": [{**{k: v for k, v in p.items() if k != "T"},
                       "T": p["T"].tolist(), **s} for p, s in zip(pts, sol)],
        "ik_success": len(good), "ik_requested": len(sol),
        "dt_s": DT_S, "duration_s": round(len(pts) * DT_S, 3),
    }
    if a.out:
        Path(a.out).expanduser().write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"리포트 -> {a.out}")

    if a.out_replay:
        if len(good) != len(sol):
            raise SystemExit(f"!! IK 가 {len(sol)-len(good)}개 거부됐다. "
                             "부분 궤적을 내보내지 않는다")
        if cfg.get("status") != "measured":
            raise SystemExit("!! 예시 좌표로는 재생 파일을 내보내지 않는다. "
                             "status: measured 가 된 뒤에 다시 하라")
        rows = [[*s["q"], p["gap_m"]] for s, p in zip(good, pts)]
        Path(a.out_replay).expanduser().write_text(json.dumps(rows), encoding="utf-8")
        print(f"재생용 궤적 -> {a.out_replay}  ({len(rows)} × 6열)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
