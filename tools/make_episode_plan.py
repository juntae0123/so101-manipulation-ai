#!/usr/bin/env python3
"""Build the 02 builder's episode plan from gate results, not from a directory listing.
02 빌더가 먹는 episode plan 을 **게이트 결과**로 만든다. 디렉터리 목록으로 만들지 않는다.

왜 필요한가 🟢
--------------
2026-09-21 실증 — raw 92편을 그대로 plan 에 넣고 `build_dataset.py build` 를 돌렸더니
한 편에서 `ValueError: marker missing run 17 exceeds limit` 로 **전체가 중단**됐다.
빌더는 불량 편을 건너뛰지 않는다. 그래서 plan **앞에** 게이트가 있어야 한다.
현석이 92편 중 76편으로 v4 를 만든 것도 같은 이유로 plan 을 미리 걸렀기 때문이다.

plan 스키마 (02_umi_dataset_builder/src/umi_dataset/replay_buffer.py 로 대조함)
    {"schema_version": 1,
     "slam_tag": "<tx_slam_tag.json>",              # 공유 태그 하나
     "episodes": [{"session": ..., "trajectory": ..., "gripper": ...}, ...]}
  - 경로는 plan 파일 위치 기준으로 해석된다 (여기서는 절대경로로 쓴다)
  - 편마다 slam_tag 를 주는 것도 되지만 **섞으면 빌더가 거부한다**

Usage
    python make_episode_plan.py --selftest
    python make_episode_plan.py --gate out/slam_gate.json --raw ~/raw --processed ~/processed \
        --slam-tag ~/processed_mapping/rec_x/tx_slam_tag.json --out out/plan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAJ = ("slam", "camera_trajectory.csv")
GRIP = ("gripper_width.csv",)
VIDEO = ("video.mp4",)


def passing_episodes(gate: dict) -> tuple[list[str], dict]:
    """Episode names the gate passed, with the tally. 게이트가 통과시킨 편과 모수."""
    eps = gate.get("episodes", [])
    names, tally = [], {"PASS": 0, "FAIL": 0, "INCOMPLETE": 0}
    for e in eps:
        v = (e.get("gates") or {}).get("verdict")
        tally[v] = tally.get(v, 0) + 1
        if v == "PASS":
            names.append(e["episode"])
    return names, {"tally": tally, "total": len(eps)}


def resolve(names: list[str], raw: Path, proc: Path) -> tuple[list[dict], list[dict]]:
    """Map names to the three files the builder needs. 필요한 파일 3종을 잇는다."""
    eps, missing = [], []
    for n in names:
        s = raw / n
        t = proc / n / TRAJ[0] / TRAJ[1]
        g = proc / n / GRIP[0]
        v = s / VIDEO[0]
        gone = [k for k, p in (("video", v), ("trajectory", t), ("gripper", g)) if not p.exists()]
        if gone:
            missing.append({"episode": n, "missing": gone})
        else:
            eps.append({"session": str(s), "trajectory": str(t), "gripper": str(g)})
    return eps, missing


def build_plan(gate: dict, raw: Path, proc: Path, tag: Path) -> dict:
    """Assemble the plan document. plan 문서를 만든다. 모수를 provenance 에 남긴다."""
    names, info = passing_episodes(gate)
    eps, missing = resolve(names, raw, proc)
    return {
        "schema_version": 1,
        "slam_tag": str(tag),
        "episodes": eps,
        "provenance": {
            "gate_verdicts": info["tally"],
            "gate_total": info["total"],
            "gate_passed": len(names),
            "plan_episodes": len(eps),
            "dropped_missing_files": missing,
            "note": ("게이트 통과 편만 담는다. 빌더는 불량 편에서 전체를 중단하므로 "
                     "여기서 거르지 않으면 build 가 죽는다 (2026-09-21 실증)"),
        },
    }


def _gate(names_verdicts: list[tuple[str, str]]) -> dict:
    """Synthetic gate doc. 정답을 아는 게이트 문서."""
    return {"episodes": [{"episode": n, "gates": {"verdict": v}} for n, v in names_verdicts],
            "total": len(names_verdicts)}


def _tree(root: Path, names: list[str], *, drop_traj: set[str] = frozenset()) -> tuple[Path, Path]:
    """Fake raw/processed trees. 정답을 아는 입력 트리."""
    raw, proc = root / "raw", root / "processed"
    for n in names:
        (raw / n).mkdir(parents=True, exist_ok=True)
        (raw / n / "video.mp4").write_bytes(b"\x00")
        (proc / n / "slam").mkdir(parents=True, exist_ok=True)
        (proc / n / "gripper_width.csv").write_text("x\n", encoding="utf-8")
        if n not in drop_traj:
            (proc / n / "slam" / "camera_trajectory.csv").write_text("x\n", encoding="utf-8")
    return raw, proc


def selftest() -> int:
    """Known-answer and discriminating rows. 정답 아는 행과 판별행."""
    import tempfile
    log, bad = [], 0

    def chk(n, c, note=""):
        nonlocal bad
        log.append((n, bool(c), note))
        if not c:
            bad += 1

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        names = ["rec_a", "rec_b", "rec_c", "rec_d"]
        raw, proc = _tree(root, names)
        tag = root / "tx_slam_tag.json"; tag.write_text("{}", encoding="utf-8")

        g = _gate([("rec_a", "PASS"), ("rec_b", "FAIL"),
                   ("rec_c", "PASS"), ("rec_d", "INCOMPLETE")])
        p = build_plan(g, raw, proc, tag)
        chk("1 통과 편만 담는다", len(p["episodes"]) == 2,
            f"담김 {len(p['episodes'])} / 게이트 전체 {p['provenance']['gate_total']}")
        chk("2 불합격·미판정은 빠진다 (판별행)",
            all("rec_b" not in e["session"] and "rec_d" not in e["session"] for e in p["episodes"]))
        chk("3 모수 기록", p["provenance"]["gate_verdicts"] == {"PASS": 2, "FAIL": 1, "INCOMPLETE": 1},
            str(p["provenance"]["gate_verdicts"]))
        chk("4 공유 slam_tag 하나", p["slam_tag"] == str(tag) and
            all("slam_tag" not in e for e in p["episodes"]), "편별 태그를 섞으면 빌더가 거부한다")
        chk("5 스키마 버전", p["schema_version"] == 1)

        raw2, proc2 = _tree(root / "b", names, drop_traj={"rec_c"})
        p2 = build_plan(g, raw2, proc2, tag)
        chk("6 파일 결측 편은 빠지고 사유가 남는다 (판별행)",
            len(p2["episodes"]) == 1 and p2["provenance"]["dropped_missing_files"][0]["missing"] == ["trajectory"],
            f"담김 {len(p2['episodes'])} · 결측 {p2['provenance']['dropped_missing_files']}")

        p3 = build_plan(_gate([]), raw, proc, tag)
        chk("7 게이트가 비면 plan 도 빈다 (통과 아님)",
            p3["episodes"] == [] and p3["provenance"]["gate_total"] == 0)

        g4 = _gate([(n, "FAIL") for n in names])
        chk("8 전부 불합격 -> 0편", len(build_plan(g4, raw, proc, tag)["episodes"]) == 0)

    for nm, o, note in log:
        print(f"  {'OK ' if o else 'FAIL'}  {nm}" + (f"   {note}" if note else ""))
    print(f"\n자체검증 {len(log) - bad}/{len(log)}")
    return 1 if bad else 0


def main() -> None:
    """CLI entry point. 명령행 진입점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gate", help="gate_slam_batch.py --out 산출물")
    ap.add_argument("--raw", help="rec_* 원본 디렉터리")
    ap.add_argument("--processed", help="rec_* SLAM 산출 디렉터리")
    ap.add_argument("--slam-tag", help="tx_slam_tag.json")
    ap.add_argument("--out", help="plan json 경로")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. plan 을 내지 않는다")
    for need in ("gate", "raw", "processed", "slam_tag", "out"):
        if not getattr(a, need):
            ap.error(f"--{need.replace('_', '-')} 가 필요하다")

    tag = Path(a.slam_tag).expanduser()
    if not tag.exists():
        raise SystemExit(f"!! slam_tag 가 없다: {tag}")
    gate = json.loads(Path(a.gate).expanduser().read_text(encoding="utf-8"))
    plan = build_plan(gate, Path(a.raw).expanduser(), Path(a.processed).expanduser(), tag)
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    pr = plan["provenance"]
    print(f"\n게이트 {pr['gate_verdicts']} / 전체 {pr['gate_total']}")
    print(f"plan 에 담긴 편 {pr['plan_episodes']} · 파일 결측으로 빠진 편 {len(pr['dropped_missing_files'])}")
    for d in pr["dropped_missing_files"][:5]:
        print(f"  결측 {d['episode']}: {d['missing']}")
    if not plan["episodes"]:
        raise SystemExit("!! plan 이 0편이다. 게이트 결과와 경로를 먼저 봐라")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
