"""How far can the camera mount be off before the policy stops working?
카메라 마운트가 얼마나 틀어져도 정책이 버티는가.

왜 우리가 만드나 / Why we build this instead of waiting
------------------------------------------------------
HW 요청 1번(카메라 마운트 도면값)은 아직 못 받았다. 그런데 **허용치는 우리가
잴 수 있다.** 도면값을 기다리는 대신 "각도 ±X도 · 높이 ±Y mm 이내로 만들어
주세요" 라는 **사양**을 만들어 되돌려준다.

이게 급한 이유 — 카메라가 5도 틀어지면 작업거리 0.25m 에서 겉보기 21.9mm 다.
파지 여유가 ±25mm 니까 여유를 거의 다 먹는다. 그리고 **이런 상수 편향은 학습이
지우지 못한다** (±10mm 에서 재생 3/4 🟢).

무엇을 흔드나 / What is perturbed
---------------------------------
```
각도  카메라 자체 x축 둘레 기울기 (마운트 pitch)   0 / ±2 / ±5 / ±10 도
높이  손 좌표계 z 방향 장착 높이                   0 / ±5 / ±10 / ±20 mm
```
정책은 **재학습하지 않는다.** 기존 E1 체크포인트를 그대로 쓴다.
즉 이 숫자는 "학습 때와 다른 카메라를 만났을 때의 열화" 다.

계측기가 스스로를 못 믿는 지점 / Where this refuses itself
----------------------------------------------------------
- **정답 아는 행**: 0도·0mm 조건은 교란 없는 기준선과 **같은 시드에서 같은 값**이
  나와야 한다. 다르면 배선이 틀렸다
- **판별력 행**: 일부러 크게 틀어 놓은 조건(기본 45도)이 성공률을 떨어뜨려야 한다.
  안 떨어지면 `camera_path` 가 **읽히지 않는 것**이고, 그러면 모든 조건이
  "영향 없음"으로 공짜로 나온다. 이게 이 프로젝트에서 9번 반복된 실패 모양이다
- **모수를 같이 찍는다**: 조건 수 · 완료 수 · 조건당 에피소드 수

인자 대조 🟢
-----------
`simulation/evaluate.py` 소스에서 직접 확인했다 —
`--checkpoint --output --seed --episodes --seconds --action-steps --grip-preload
 --task --image-perturb --image-perturb-sigma`. `--camera` 는 **없다.**
그래서 교란은 task yaml 의 `camera_path:` 로 넣는다
(`simulation/env.py:14` 가 `task.get('camera_path','configs/camera.yaml')` 를 읽는다).

Usage
-----
  # [서버]
  ~/envs/handoff312/bin/python AI/tools/sweep_camera_mount.py --selftest
  ~/envs/handoff312/bin/python AI/tools/sweep_camera_mount.py --handoff ~/handoff --checkpoint ~/handoff/outputs/e1/checkpoints/latest.ckpt --plan   # 설정 생성 + GPU별 잡 파일
  ~/envs/handoff312/bin/python AI/tools/sweep_camera_mount.py --handoff ~/handoff --collect   # 끝난 뒤 집계

되돌리기 / Reverting
--------------------
생성물은 전부 `<handoff>/configs/_sweep_camera/` 와 `<handoff>/outputs/sweep_camera/`
안에만 들어간다. 두 디렉터리를 지우면 원상복구다. 기존 설정은 **읽기만** 한다.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

GPUS = (1, 2, 3, 4, 6)          # 지침: 0·5·7·8·9 는 타인 것
ANGLES_DEG = (0.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0)
HEIGHTS_MM = (5.0, -5.0, 10.0, -10.0, 20.0, -20.0)
DISCRIMINATING_DEG = 45.0
SWEEP_CFG = "configs/_sweep_camera"
SWEEP_OUT = "outputs/sweep_camera"


# ⚠️ 2026-09-19 정정 — 초판 잡 파일에 `MUJOCO_GL=egl` 이 없었다. 서버에 DISPLAY 가
#    없으므로 렌더러가 즉시 죽고 5개 GPU 잡이 전부 exit 1 났다. evaluate.py 의
#    **인자**만 소스로 대조하고 **환경변수**는 안 봤다. 기존 러너 8종이 전부
#    이 줄을 갖고 있었다 (run_e1_train.sh:61 · run_e2.sh:22 · run_e2_folds.sh:82 …).
JOB_HEADER = """set -eu
export MUJOCO_GL=egl
cd {handoff}
# 렌더링이 되는지 먼저 확인한다. 20편 돌린 뒤에 죽는 것보다 여기서 죽는 게 싸다.
python -c "import os,sys; sys.exit(0 if os.environ.get('MUJOCO_GL')=='egl' else 1)" \\
  || {{ echo '!! MUJOCO_GL 이 egl 이 아니다'; exit 2; }}
"""


def rx(deg: float) -> np.ndarray:
    """Rotation about x as 4x4. x축 회전 4x4."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return T


def perturb(T: np.ndarray, angle_deg: float, height_mm: float) -> np.ndarray:
    """Tilt about the camera's own x, then shift along the hand's z.
    카메라 자체 x 둘레로 기울이고, 손 좌표계 z 로 옮긴다.

    순서가 중요하다 — `T @ rx` 는 카메라 축 기준 회전이고 `rx @ T` 는 손 축
    기준이다. 마운트가 틀어지는 것은 카메라 축 쪽이다."""
    out = np.asarray(T, dtype=np.float64) @ rx(angle_deg)
    out[2, 3] += height_mm / 1000.0
    return out


def conditions(include_disc: bool = True) -> list[dict]:
    """Every condition with a stable name. 조건 전체. 이름은 고정."""
    c = [{"name": f"a{a:+.0f}".replace("+0", "0"), "angle": a, "height": 0.0}
         for a in ANGLES_DEG]
    c += [{"name": f"z{h:+.0f}", "angle": 0.0, "height": h} for h in HEIGHTS_MM]
    if include_disc:
        c.append({"name": "DISC", "angle": DISCRIMINATING_DEG, "height": 0.0})
    return c


def write_configs(handoff: Path, task_name: str, conds: list[dict]) -> list[dict]:
    """Write one camera yaml + one task yaml per condition. Originals untouched.
    조건마다 카메라 yaml 과 task yaml 을 하나씩 쓴다. 원본은 안 건드린다."""
    cam_src = handoff / "configs" / "camera.yaml"
    task_src = handoff / "configs" / task_name
    for p in (cam_src, task_src):
        if not p.exists():
            raise SystemExit(f"!! 없다: {p}")
    cam = yaml.safe_load(cam_src.read_text(encoding="utf-8"))
    task = yaml.safe_load(task_src.read_text(encoding="utf-8"))
    T0 = np.asarray(cam["T_hand_camera"], dtype=np.float64)

    out_dir = handoff / SWEEP_CFG
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for c in conds:
        cam_c = dict(cam)
        cam_c["T_hand_camera"] = perturb(T0, c["angle"], c["height"]).tolist()
        cam_c["calibration_status"] = (
            f"sweep_perturbation angle={c['angle']}deg height={c['height']}mm "
            "— 교란본이다. 실제 캘리브레이션 아님")
        cam_p = out_dir / f"camera_{c['name']}.yaml"
        cam_p.write_text(yaml.safe_dump(cam_c, sort_keys=False, allow_unicode=True),
                         encoding="utf-8")

        task_c = dict(task)
        task_c["camera_path"] = f"{SWEEP_CFG}/camera_{c['name']}.yaml"
        task_p = out_dir / f"task_{c['name']}.yaml"
        task_p.write_text(yaml.safe_dump(task_c, sort_keys=False, allow_unicode=True),
                          encoding="utf-8")
        made.append({**c, "task": str(task_p), "camera": str(cam_p)})
    return made


def selftest() -> int:
    """Known answers and deliberately wrong inputs. 정답 아는 행 + 고의 오답."""
    ok = tot = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok, tot
        tot += 1
        ok += bool(cond)
        print(f"  {'OK ' if cond else '!! '} [{tot}] {label:<46} {detail}")

    T = np.eye(4)
    T[:3, 3] = [0.0, 0.025, 0.085]
    T[:3, :3] = rx(30.0)[:3, :3]

    check("0도 0mm 는 항등 (정답 아는 행)",
          np.allclose(perturb(T, 0.0, 0.0), T, atol=1e-15))
    p5 = perturb(T, 5.0, 0.0)
    check("5도는 회전만 바꾼다", np.allclose(p5[:3, 3], T[:3, 3], atol=1e-15))
    ang = math.degrees(math.acos(
        max(-1.0, min(1.0, (np.trace(T[:3, :3].T @ p5[:3, :3]) - 1) / 2))))
    check("5도 회전의 측지각이 5도", abs(ang - 5.0) < 1e-9, f"{ang:.9f}도")
    p20 = perturb(T, 0.0, 20.0)
    check("20mm 는 z 만 20mm 올린다",
          abs(p20[2, 3] - (T[2, 3] + 0.020)) < 1e-15
          and np.allclose(p20[:3, :3], T[:3, :3], atol=1e-15))

    # 판별력: 카메라 축 회전과 손 축 회전은 달라야 한다. 같으면 순서가 무의미해진다.
    check("판별력: T@Rx 와 Rx@T 가 다르다",
          not np.allclose(T @ rx(5.0), rx(5.0) @ T, atol=1e-9))
    # 판별력: 교란이 실제로 뭔가 바꿔야 한다.
    check("판별력: 5도와 -5도가 다르다",
          not np.allclose(perturb(T, 5.0, 0.0), perturb(T, -5.0, 0.0), atol=1e-9))

    cs = conditions()
    names = [c["name"] for c in cs]
    check(f"조건 수 {len(cs)} = 각도 7 + 높이 6 + 판별 1", len(cs) == 14)
    check("조건 이름이 전부 다르다", len(set(names)) == len(names))
    check("판별 조건이 들어 있다", "DISC" in names)
    check("판별 조건을 뺄 수 있다", len(conditions(False)) == 13)
    check("GPU 는 허용된 것만", set(GPUS) == {1, 2, 3, 4, 6})
    hdr = JOB_HEADER.format(handoff="/x")
    check("잡 헤더에 MUJOCO_GL=egl 이 있다 (정답 아는 행)",
          "export MUJOCO_GL=egl" in hdr)
    check("판별력: 헤더에서 그 줄을 빼면 검사가 걸린다",
          "export MUJOCO_GL=egl" not in hdr.replace("export MUJOCO_GL=egl", ""))
    check("잡 헤더가 handoff 경로로 cd 한다", "cd /x" in hdr)

    print(f"자체검증 {ok} / {tot}")
    return 0 if ok == tot else 1


def cmd_plan(a) -> int:
    handoff = Path(a.handoff).expanduser()
    conds = write_configs(handoff, a.task_name, conditions())
    out_root = handoff / SWEEP_OUT
    out_root.mkdir(parents=True, exist_ok=True)

    py = a.python
    jobs: dict[int, list[str]] = {g: [] for g in GPUS}
    for i, c in enumerate(conds):
        g = GPUS[i % len(GPUS)]
        o = out_root / c["name"]
        jobs[g].append(
            f"CUDA_VISIBLE_DEVICES={g} {py} simulation/evaluate.py "
            f"--checkpoint {a.checkpoint} --task {c['task']} --output {o} "
            f"--seed {a.seed} --episodes {a.episodes} --action-steps {a.action_steps}")

    for g, lines in jobs.items():
        f = out_root / f"jobs_gpu{g}.sh"
        f.write_text(JOB_HEADER.format(handoff=handoff) + "\n".join(lines)
                     + f"\necho SWEEPCAM_GPU{g}_DONE\n", encoding="utf-8")
        print(f"GPU {g}: {len(lines)}개 조건 → {f}")

    print(f"\n조건 {len(conds)}개 · 조건당 {a.episodes}편 · "
          f"총 {len(conds)*a.episodes} 롤아웃 (약 {len(conds)*a.episodes*24.6/60/len(GPUS):.0f}분, GPU {len(GPUS)}장 병렬)")
    print("\n실행 —")
    for g in GPUS:
        print(f"  nohup bash {out_root}/jobs_gpu{g}.sh > {out_root}/gpu{g}.log 2>&1 &")
    print(f"\n완료 확인:  grep -c SWEEPCAM_GPU._DONE {out_root}/gpu*.log")
    print(f"집계:      {py} AI/tools/sweep_camera_mount.py --handoff {handoff} --collect")
    return 0


def cmd_collect(a) -> int:
    root = Path(a.handoff).expanduser() / SWEEP_OUT
    conds = conditions()
    rows, missing = [], []
    for c in conds:
        f = root / c["name"] / "evaluation.json"
        if not f.exists():
            missing.append(c["name"])
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        n = len(d["episodes"])
        s = sum(bool(e["success"]) for e in d["episodes"])
        rows.append({**c, "success": s, "episodes": n,
                     "rate": s / n if n else float("nan")})
    print(f"집계 {len(rows)} / 조건 {len(conds)}" +
          (f" · 없음 {missing}" if missing else ""))
    if not rows:
        raise SystemExit("!! 결과가 0개다. 잡이 돌았는지 로그부터 본다")

    by = {r["name"]: r for r in rows}
    if "a0" not in by:
        raise SystemExit("!! 기준선(a0) 결과가 없다. 기준선 없이 열화를 말하지 않는다")
    base = by["a0"]["rate"]
    print(f"\n기준선 a0 (0도 0mm): {by['a0']['success']} / {by['a0']['episodes']} "
          f"= {base*100:.1f}%")

    if "DISC" in by:
        d = by["DISC"]
        print(f"판별력 {DISCRIMINATING_DEG:.0f}도: {d['success']} / {d['episodes']} "
              f"= {d['rate']*100:.1f}%")
        if d["rate"] >= base - 1e-9:
            raise SystemExit(
                f"!! {DISCRIMINATING_DEG:.0f}도 틀어도 기준선과 같거나 낫다.\n"
                "   camera_path 가 안 읽히는 것일 가능성이 높다. 이 상태로는 모든\n"
                "   조건이 '영향 없음'으로 공짜로 나온다. 허용치를 내지 않는다")
    else:
        print("⚠️ 판별 조건이 없다. 이 스윕의 결과를 사양으로 쓰지 마라")

    print(f"\n{'조건':<8}{'각도':>7}{'높이mm':>8}{'성공':>10}{'성공률':>9}{'기준선차':>10}")
    for r in sorted(rows, key=lambda r: (r["name"] == "DISC", r["angle"], r["height"])):
        print(f"{r['name']:<8}{r['angle']:>7.0f}{r['height']:>8.0f}"
              f"{r['success']:>6}/{r['episodes']:<3}{r['rate']*100:>8.1f}%"
              f"{(r['rate']-base)*100:>+9.1f}%p")

    # 사양: 기준선 대비 낙폭이 --drop 이내인 가장 큰 |각도| · |높이|
    def tol(key: str) -> float:
        # `== 0.0` / `!= 0.0` 는 의도한 동등 비교다. 이 값들은 계산 결과가 아니라
        # ANGLES_DEG · HEIGHTS_MM 에 우리가 리터럴 0.0 으로 박아 넣은 것이라
        # 부동소수 오차가 낄 자리가 없다.
        good = [abs(r[key]) for r in rows
                if r["name"] != "DISC" and r[key] != 0.0
                and (base - r["rate"]) * 100 <= a.drop
                and (r["height"] if key == "angle" else r["angle"]) == 0.0]
        return max(good) if good else 0.0

    spec = {"angle_deg": tol("angle"), "height_mm": tol("height")}
    print(f"\n사양 (기준선 대비 낙폭 {a.drop}%p 이내):")
    print(f"  마운트 각도  ±{spec['angle_deg']:.0f}도 이내")
    print(f"  마운트 높이  ±{spec['height_mm']:.0f} mm 이내")
    if spec["angle_deg"] == 0.0 or spec["height_mm"] == 0.0:
        print("  ⚠️ 0 이 나왔다 — 가장 작은 교란도 낙폭을 넘었다는 뜻이거나,\n"
              "     그 방향 조건이 통째로 빠졌다는 뜻이다. 표에서 어느 쪽인지 먼저 봐라")
    print("\n⚠️ 시뮬 수치다. 조건 병기 없이 인용 금지. 재학습 없이 기존 정책을 쓴 결과이므로\n"
          "   '이 정도 틀어져도 된다'는 **현재 정책 기준**이고 도메인 랜덤화를 넣으면 달라진다")

    out = {"baseline_rate": base, "drop_budget_pp": a.drop, "rows": rows,
           "missing": missing, "spec": spec,
           "_conditions": f"handoff 시뮬 · 조건당 {rows[0]['episodes']}편 · 재학습 없음"}
    (root / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"→ {root/'summary.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", action="store_true", help="설정 생성 + GPU별 잡 파일")
    ap.add_argument("--collect", action="store_true", help="끝난 뒤 집계")
    ap.add_argument("--handoff", default="~/handoff")
    ap.add_argument("--checkpoint")
    ap.add_argument("--task-name", default="can_side.yaml")
    ap.add_argument("--python", default="~/envs/handoff312/bin/python")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--action-steps", type=int, default=4)
    ap.add_argument("--drop", type=float, default=10.0,
                    help="사양 판정: 기준선 대비 허용 낙폭 [%%p]")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패")
    print()
    if a.plan:
        if not a.checkpoint:
            raise SystemExit("!! --checkpoint 가 필요하다")
        return cmd_plan(a)
    if a.collect:
        return cmd_collect(a)
    raise SystemExit("!! --plan / --collect / --selftest 중 하나")


if __name__ == "__main__":
    raise SystemExit(main())
