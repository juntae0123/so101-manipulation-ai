"""Sample camera configs for domain randomization, and the task yamls that use them.
도메인 랜덤화용 카메라 설정과 그것을 물리는 task yaml 을 만든다.

사전등록: AI/docs/PREREG_domain_randomization_0919.md (실행 전 작성)

무엇을 만드나
-------------
`<handoff>/configs/_domran/` 안에만 만든다. 기존 설정은 **읽기만** 한다.
```
camera_g{i}.yaml   T_hand_camera @ Rx(theta_i) + z_i     교란본
task_g{i}.yaml     can_side.yaml + camera_path 만 추가
```
교란식은 `sweep_camera_mount.perturb` 를 **그대로 재사용**한다. 두 벌이면 갈린다.

Usage
-----
  # [서버]
  ~/envs/handoff312/bin/python AI/tools/make_domran_cameras.py --selftest
  ~/envs/handoff312/bin/python AI/tools/make_domran_cameras.py --handoff ~/handoff --groups 10

되돌리기
--------
`<handoff>/configs/_domran/` 디렉터리를 지우면 원상복구다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_camera_mount import perturb  # noqa: E402 — 교란식을 재사용한다

ANGLE_MAX_DEG = 5.0
HEIGHT_MAX_MM = 10.0
SAMPLE_SEED = 20260919          # 결과와 무관하게 고정. 실행 전에 박았다
OUT_DIR = "configs/_domran"


def sample(groups: int, seed: int = SAMPLE_SEED) -> list[dict]:
    """Deterministic draws. 같은 시드면 항상 같은 표본.

    균등 추출이되 **공칭(0,0)을 반드시 한 칸 포함한다** — 공칭이 빠지면
    '교란만 배우고 정상은 못 하는' 모델이 나와도 G1 으로 못 잡는다."""
    rng = np.random.default_rng(seed)
    out = [{"name": "g0", "angle": 0.0, "height": 0.0}]
    for i in range(1, groups):
        out.append({"name": f"g{i}",
                    "angle": float(rng.uniform(-ANGLE_MAX_DEG, ANGLE_MAX_DEG)),
                    "height": float(rng.uniform(-HEIGHT_MAX_MM, HEIGHT_MAX_MM))})
    return out


def write(handoff: Path, task_name: str, groups: list[dict],
          episodes_per_group: int, seed0: int) -> list[dict]:
    """One camera yaml + one task yaml per group. Originals untouched."""
    cam_src = handoff / "configs" / "camera.yaml"
    task_src = handoff / "configs" / task_name
    for p in (cam_src, task_src):
        if not p.exists():
            raise SystemExit(f"!! 없다: {p}")
    cam = yaml.safe_load(cam_src.read_text(encoding="utf-8"))
    task = yaml.safe_load(task_src.read_text(encoding="utf-8"))
    T0 = np.asarray(cam["T_hand_camera"], dtype=np.float64)

    d = handoff / OUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    made = []
    for i, g in enumerate(groups):
        c = dict(cam)
        c["T_hand_camera"] = perturb(T0, g["angle"], g["height"]).tolist()
        c["calibration_status"] = (f"domain_randomization angle={g['angle']:.4f}deg "
                                   f"height={g['height']:.4f}mm — 교란본이다")
        (d / f"camera_{g['name']}.yaml").write_text(
            yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
        t = dict(task)
        t["camera_path"] = f"{OUT_DIR}/camera_{g['name']}.yaml"
        (d / f"task_{g['name']}.yaml").write_text(
            yaml.safe_dump(t, sort_keys=False, allow_unicode=True), encoding="utf-8")
        made.append({**g, "seed": seed0 + i * episodes_per_group,
                     "episodes": episodes_per_group,
                     "task": str(d / f"task_{g['name']}.yaml")})
    (d / "plan.json").write_text(json.dumps(
        {"_PREREG": "AI/docs/PREREG_domain_randomization_0919.md",
         "sample_seed": SAMPLE_SEED, "angle_max_deg": ANGLE_MAX_DEG,
         "height_max_mm": HEIGHT_MAX_MM, "groups": made},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return made


def selftest() -> int:
    ok = tot = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok, tot
        tot += 1
        ok += bool(cond)
        print(f"  {'OK ' if cond else '!! '} [{tot}] {label:<46} {detail}")

    g = sample(10)
    check("그룹 수가 모수와 맞는다", len(g) == 10)
    # 리터럴 0.0 을 우리가 직접 넣었으므로 동등 비교가 의도다 (계산 결과가 아니다)
    check("g0 은 공칭(0, 0) 이다 (정답 아는 행)",
          g[0]["angle"] == 0.0 and g[0]["height"] == 0.0)
    check("나머지는 범위 안", all(abs(x["angle"]) <= ANGLE_MAX_DEG
                             and abs(x["height"]) <= HEIGHT_MAX_MM for x in g[1:]))
    check("이름이 전부 다르다", len({x["name"] for x in g}) == len(g))
    check("같은 시드면 같은 표본 (재현)", sample(10) == g)
    check("판별력: 시드가 다르면 표본도 다르다", sample(10, SAMPLE_SEED + 1) != g)
    # 위와 같은 이유로 동등 비교가 의도다
    check("판별력: 교란이 0 인 그룹이 g0 하나뿐",
          sum(1 for x in g if x["angle"] == 0.0 and x["height"] == 0.0) == 1)

    T = np.eye(4)
    T[:3, 3] = [0.0, 0.025, 0.085]
    check("교란식 재사용: 0도 0mm 는 항등",
          np.allclose(perturb(T, 0.0, 0.0), T, atol=1e-15))

    print(f"자체검증 {ok} / {tot}")
    return 0 if ok == tot else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--handoff", default="~/handoff")
    ap.add_argument("--task-name", default="can_side.yaml")
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--episodes-per-group", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=6000)
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패")
    print()

    made = write(Path(a.handoff).expanduser(), a.task_name,
                 sample(a.groups), a.episodes_per_group, a.seed0)
    total = sum(m["episodes"] for m in made)
    print(f"설정 {len(made)} / 기대 {a.groups} · 총 편수 {total} / 기대 "
          f"{a.groups * a.episodes_per_group}")
    for m in made:
        print(f"  {m['name']:<4} 각도 {m['angle']:+7.3f}도 · 높이 {m['height']:+7.3f}mm "
              f"· seed {m['seed']}~{m['seed'] + m['episodes'] - 1}")
    print(f"\n→ {Path(a.handoff).expanduser() / OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
