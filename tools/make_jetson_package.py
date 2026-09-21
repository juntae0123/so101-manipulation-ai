#!/usr/bin/env python3
"""One command: training run -> verified Jetson package. 학습 결과 -> 검증된 젯슨 패키지.

ckpt 를 갈아끼우며 반복하는 자리다. 매번 손으로 폴더 만들고 report 옮기고 인자
맞추면 어딘가 틀린다. 그 과정을 한 줄로 묶고 **끝에 게이트를 건다.**

무엇을 하나
-----------
1. 프리플라이트 — 필요한 파일이 전부 있는지. 없으면 여기서 멈춘다
2. 작업 폴더를 **새로 만든다** — `*.report.json` 이 정확히 하나여야 하고
   (`export_so101_pick_v1.py:225` 가 2개 이상이면 죽는다), 남은 찌꺼기가 섞이면 안 된다
3. 변환기 실행 (로그는 실시간으로 흐른다)
4. **산출물 게이트** — 필수 4파일 · pc_check PASS · metadata 계약 · dataset.report 상태
   · reference.npz 액션 규모. 하나라도 어긋나면 **tgz 를 내보내지 않는다**

왜 게이트가 필요한가 🟢
-----------------------
2026-09-21 실증 — 경량 ckpt(`pickles.epoch` 없음)를 넣었더니 `encoder.pt`·`denoiser.pt`·
`reference.npz` 까지만 쓰이고 `metadata.json` 부터 안 써진 폴더가 남았다.
**그 폴더는 겉보기에 정상이다.** 파일이 세 개 있고 에러 메시지는 스크롤 위로 사라진다.
젯슨에 그걸 올리면 실행기가 metadata 를 못 찾는다.

⚠️ 이 도구가 말하지 않는 것
    변환 성공은 **실물 작업 성공률이 아니다.** pc_check PASS 는 TorchScript 출력이
    원본 파이토치와 같다는 뜻이지 로봇이 물체를 집는다는 뜻이 아니다.

Usage
    python make_jetson_package.py --selftest
    python make_jetson_package.py --run-dir out/atlas_v1_ours/runs/atlas_v1_ours_154657 \
        --dataset-report ~/hyeonseok/umi_gpu_bundle/data/<batch>/<batch>.zarr.report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED = ("encoder.pt", "denoiser.pt", "metadata.json", "dataset.report.json")
KEPT = ("config.yaml", "reference.npz", "pc_check.json")
META_KEYS = ("shape_meta", "scheduler", "inference_steps", "inference_image_transform",
             "action_scale", "action_offset", "checkpoint_sha256", "checkpoint_epoch")
DEF_EXPORTER = "~/hyeonseok/Untitled Folder/export_so101_pick_v1.py"
DEF_UMI = "~/hyeonseok/umi_gpu_bundle/third_party/umi"
# 상대 액션이면 청크 안 이동이 이 안쪽이어야 한다. 넘으면 절대 좌표를 의심한다.
REL_POS_MAX_MM = 300.0


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def run(cmd, cwd=None, title=None) -> int:
    if title:
        print(f"\n── {title}\n$ {' '.join(map(str, cmd))}\n", flush=True)
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd and str(cwd),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    print(f"[종료코드 {p.returncode}]", flush=True)
    return p.returncode


def action_scale_report(out: Path) -> dict:
    """reference.npz 의 액션 규모. 상대인지 절대인지의 유일한 실측 단서다."""
    try:
        import numpy as np
    except ImportError:
        return {"ok": None, "why": "numpy 없음 — 대조 불가"}
    p = out / "reference.npz"
    if not p.exists():
        return {"ok": None, "why": "reference.npz 없음 — 대조 불가"}
    try:
        z = np.load(p, allow_pickle=False)
    except Exception as exc:                      # noqa: BLE001
        # 못 읽은 것과 규모가 괜찮은 것을 가른다. 조용히 통과시키지 않는다
        return {"ok": None, "why": f"reference.npz 를 못 읽었다 ({type(exc).__name__}) — 대조 불가"}
    best = None
    for k in z:
        a = z[k]
        if a.ndim >= 2 and a.shape[-1] == 10:
            best = (k, a.reshape(-1, 10))
            break
    if best is None:
        return {"ok": None, "why": f"(*,10) 배열이 없다. 키 {list(z)} — 대조 불가"}
    k, x = best
    mm = float(abs(x[:, :3]).max()) * 1000.0
    return {"ok": mm <= REL_POS_MAX_MM, "key": k, "pos_abs_max_mm": mm,
            "gap_min_m": float(x[:, 9].min()), "gap_max_m": float(x[:, 9].max()),
            "why": (f"pos |최대| {mm:.1f} mm (기준 <= {REL_POS_MAX_MM}). "
                    "넘으면 절대 좌표일 수 있다 — 실행기가 상대로 읽으면 틀린다")}


def verify(out: Path, archive: Path | None) -> dict:
    """산출물 게이트. 못 재면 못 잰다고 한다."""
    rows = []

    def row(name, ok, got, want):
        rows.append({"name": name, "ok": ok, "got": got, "want": want})

    have = {p.name for p in out.iterdir()} if out.is_dir() else set()
    miss = [f for f in REQUIRED if f not in have]
    row("필수 4파일", not miss, f"{len(REQUIRED) - len(miss)} / {len(REQUIRED)}"
        + (f" · 없음 {miss}" if miss else ""), "전부")
    row("보존 파일", all(f in have for f in KEPT),
        f"{sum(f in have for f in KEPT)} / {len(KEPT)}", "전부")

    pc = out / "pc_check.json"
    if pc.exists():
        d = json.loads(pc.read_text(encoding="utf-8"))
        row("pc_check 판정", d.get("status") == "PASS", str(d.get("status")), "PASS")
        row("추론 시간 기록", d.get("pc_inference_ms") is not None,
            f"{d.get('pc_inference_ms')}", "있을 것")
    else:
        row("pc_check 판정", None, "pc_check.json 없음 — 대조 불가", "PASS")
        row("추론 시간 기록", None, "대조 불가", "있을 것")

    mp = out / "metadata.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        mk = [k for k in META_KEYS if k not in m]
        row("metadata 계약 키", not mk, f"{len(META_KEYS) - len(mk)} / {len(META_KEYS)}"
            + (f" · 없음 {mk}" if mk else ""), "전부")
        sc, of = m.get("action_scale"), m.get("action_offset")
        row("action_scale/offset 길이 일치",
            isinstance(sc, list) and isinstance(of, list) and len(sc) == len(of) == 10,
            f"{len(sc) if isinstance(sc, list) else '없음'} / "
            f"{len(of) if isinstance(of, list) else '없음'}", "둘 다 10")
        # ⚠️ 값의 뜻을 우리가 모른다. 판정하지 않고 **찍기만** 한다
        rows.append({"name": "robot_execution_enabled (참고)", "ok": True,
                     "got": str(m.get("robot_execution_enabled")),
                     "want": "실행기 쪽 의미 확인 필요 — 여기선 판정하지 않는다"})
    else:
        row("metadata 계약 키", None, "metadata.json 없음 — 대조 불가", "전부")
        row("action_scale/offset 길이 일치", None, "대조 불가", "둘 다 10")

    rp = out / "dataset.report.json"
    if rp.exists():
        r = json.loads(rp.read_text(encoding="utf-8"))
        row("training_input_status", r.get("training_input_status") == "ready",
            str(r.get("training_input_status")), "ready")
        row("physical_deployment_ready", r.get("physical_deployment_ready") is True,
            str(r.get("physical_deployment_ready")), "true")
        cam = (r.get("config_sha256") or {}).get("camera_tcp")
        row("camera_tcp 해시 기록", bool(cam), (cam or "없음")[:16], "있을 것")
    else:
        for n, w in (("training_input_status", "ready"),
                     ("physical_deployment_ready", "true"),
                     ("camera_tcp 해시 기록", "있을 것")):
            row(n, None, "dataset.report.json 없음 — 대조 불가", w)

    a = action_scale_report(out)
    row("액션 규모 (상대 여부)", a.get("ok"), a.get("why"), f"pos <= {REL_POS_MAX_MM} mm")

    if archive is not None:
        row("tgz 생성", archive.exists(),
            f"{archive.stat().st_size:,} B" if archive.exists() else "없음", "있을 것")
    p = sum(1 for r in rows if r["ok"] is True)
    f = sum(1 for r in rows if r["ok"] is False)
    u = sum(1 for r in rows if r["ok"] is None)
    return {"rows": rows, "passed": p, "failed": f, "unknown": u, "total": len(rows),
            "verdict": "PASS" if f == 0 and u == 0 else ("FAIL" if f else "INCOMPLETE"),
            "action_scale": a,
            "note": "변환 성공은 실물 작업 성공률이 아니다"}


def build(run_dir: Path, dataset_report: Path, exporter: Path, umi_root: Path,
          out_root: Path, python: str) -> int:
    ck = run_dir / "checkpoints" / "best.ckpt"
    man = run_dir / "deploy"
    manifests = sorted(man.glob("*.manifest.json")) if man.is_dir() else []
    name = run_dir.name

    print("프리플라이트 —")
    ok = tot = 0
    for label, path in (("best.ckpt", ck), ("dataset.report.json", dataset_report),
                        ("변환기", exporter), ("umi-root", umi_root)):
        tot += 1; e = path.exists(); ok += e
        print(f"  [{'있음' if e else '없음'}] {label:<20} {path}")
    print(f"  manifest {len(manifests)}개 (선택)")
    print(f"\n프리플라이트 {ok} / {tot}")
    if ok != tot:
        print("!! 위 [없음] 을 먼저 해결한다")
        return 1
    print(f"변환기 sha256  {sha256(exporter)[:16]}  ← 어느 판으로 돌렸는지 남긴다")

    work = out_root / f"work_{name}"
    out = out_root / f"{name}_jetson"
    archive = work / f"{out.name}.tgz"
    if out.exists():
        print(f"!! 출력 폴더가 이미 있다: {out}\n   변환기가 FileExistsError 로 거부한다. "
              "지우거나 --out-root 를 바꿔라")
        return 1
    if work.exists():
        shutil.rmtree(work)          # report 가 2개면 변환기가 죽는다. 매번 새로 만든다
    work.mkdir(parents=True)
    shutil.copyfile(dataset_report, work / "dataset.report.json")
    for m in manifests[:1]:
        shutil.copyfile(m, work / m.name)
    reports = list(work.glob("*.report.json"))
    print(f"작업 폴더 {work}  ·  *.report.json {len(reports)}개 (1이어야 한다)")
    if len(reports) != 1:
        print("!! report 가 1개가 아니다 — 변환기가 죽거나 엉뚱한 걸 집는다")
        return 1

    env_note = "" if "hydra" in sys.modules else ""
    rc = run([python, str(exporter), "--workdir", work, "--checkpoint", ck,
              "--umi-root", umi_root, "--out", out, "--archive", archive],
             cwd=work, title=f"젯슨 변환 ({name}){env_note}")
    if rc != 0:
        print("\n!! 변환기가 실패했다. 위 로그의 마지막 예외를 본다")
        print("   `KeyError: 'epoch'` 면 경량 ckpt 를 넣은 것이다 — 원본 best.ckpt 를 써라")
        return 1

    v = verify(out, archive)
    print("\n── 산출물 게이트 ──────────────────────────────")
    for r in v["rows"]:
        mark = {True: " 통과 ", False: "불합격 ", None: "미판정 "}[r["ok"]]
        print(f"  [{mark}] {r['name']:<28} {r['got']}")
        if r["ok"] is not True:
            print(f"{'':14}기준 {r['want']}")
    print(f"\n  통과 {v['passed']} · 불합격 {v['failed']} · 미판정 {v['unknown']} / {v['total']}")
    print(f"  판정: {v['verdict']}")
    (out / "package_gate.json").write_text(json.dumps(v, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    if v["verdict"] != "PASS":
        print("\n!! 미판정은 통과가 아니다. 이 패키지를 젯슨에 올리지 마라")
        return 2

    print(f"\n젯슨에 올릴 것 —\n  {archive}\n  sha256 {sha256(archive)}")
    print(f"\n  scp '{archive}' <젯슨>:~/  &&  ssh <젯슨> "
          f"'mkdir -p models && tar -xzf ~/{archive.name} -C models && ls models/{out.name}'")
    print(f"\n** {v['note']} **")
    return 0


# ── 자체검증 (변환기·torch 없이 돈다) ────────────────────────────────────────
def _mk(d: Path, files: dict):
    d.mkdir(parents=True, exist_ok=True)
    for n, body in files.items():
        (d / n).write_text(json.dumps(body) if isinstance(body, dict) else str(body),
                           encoding="utf-8")


def selftest() -> int:
    import tempfile
    ok = tot = 0

    def chk(name, cond, note=""):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(f"  {'OK  ' if cond else '실패'} {name}   {note}")

    good_meta = {k: ([0] * 10 if k in ("action_scale", "action_offset") else 1)
                 for k in META_KEYS}
    good_meta["robot_execution_enabled"] = False
    good_rep = {"training_input_status": "ready", "physical_deployment_ready": True,
                "config_sha256": {"camera_tcp": "4e16c0a2" + "0" * 56}}

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        full = root / "full"
        _mk(full, {"encoder.pt": "x", "denoiser.pt": "x", "metadata.json": good_meta,
                   "dataset.report.json": good_rep, "config.yaml": "x",
                   "reference.npz": "x", "pc_check.json": {"status": "PASS",
                                                           "pc_inference_ms": 2741.0}})
        v = verify(full, None)
        chk("1 정상 패키지 -> 불합격 0 · npz 는 미판정 (판별행)",
            v["failed"] == 0 and v["unknown"] >= 1,
            f"{v['verdict']} · 불합격 {v['failed']} · 미판정 {v['unknown']}")

        # 오늘 실제로 나온 모양: metadata 부터 안 써진 폴더
        half = root / "half"
        _mk(half, {"encoder.pt": "x", "denoiser.pt": "x", "reference.npz": "x"})
        v2 = verify(half, None)
        chk("2 metadata 없는 폴더 -> 불합격 (오늘 실증 판별행)",
            v2["verdict"] == "FAIL" and any(r["name"] == "필수 4파일" and r["ok"] is False
                                            for r in v2["rows"]),
            f"{v2['verdict']} · 불합격 {v2['failed']}")

        fail_pc = root / "failpc"
        _mk(fail_pc, {"encoder.pt": "x", "denoiser.pt": "x", "metadata.json": good_meta,
                      "dataset.report.json": good_rep, "config.yaml": "x",
                      "reference.npz": "x", "pc_check.json": {"status": "FAIL"}})
        chk("3 pc_check FAIL -> 불합격 (판별행)", verify(fail_pc, None)["verdict"] == "FAIL")

        notready = root / "notready"
        _mk(notready, {"encoder.pt": "x", "denoiser.pt": "x", "metadata.json": good_meta,
                       "dataset.report.json": {"training_input_status": "provisional_camera_tcp",
                                               "physical_deployment_ready": False,
                                               "config_sha256": {}},
                       "config.yaml": "x", "reference.npz": "x",
                       "pc_check.json": {"status": "PASS", "pc_inference_ms": 1.0}})
        v4 = verify(notready, None)
        chk("4 physical_deployment_ready false -> 불합격 (판별행)",
            any(r["name"] == "physical_deployment_ready" and r["ok"] is False
                for r in v4["rows"]), f"불합격 {v4['failed']}")

        short = root / "short"
        m2 = dict(good_meta); m2["action_scale"] = [0] * 7
        _mk(short, {"encoder.pt": "x", "denoiser.pt": "x", "metadata.json": m2,
                    "dataset.report.json": good_rep, "config.yaml": "x",
                    "reference.npz": "x",
                    "pc_check.json": {"status": "PASS", "pc_inference_ms": 1.0}})
        chk("5 action_scale 길이 7 -> 불합격 (판별행)",
            any(r["name"].startswith("action_scale") and r["ok"] is False
                for r in verify(short, None)["rows"]))

        chk("6 robot_execution_enabled 는 판정하지 않고 찍기만 한다",
            any(r["name"].startswith("robot_execution_enabled") and r["ok"] is True
                for r in v["rows"]), "의미를 모르는 값을 게이트로 쓰지 않는다")

        try:
            import numpy as np
            npz = root / "npz"
            npz.mkdir()
            for n in ("encoder.pt", "denoiser.pt", "config.yaml"):
                (npz / n).write_text("x")
            (npz / "metadata.json").write_text(json.dumps(good_meta))
            (npz / "dataset.report.json").write_text(json.dumps(good_rep))
            (npz / "pc_check.json").write_text(json.dumps({"status": "PASS",
                                                           "pc_inference_ms": 1.0}))
            rel = np.zeros((1, 16, 10), dtype=np.float32); rel[..., :3] = 0.05
            np.savez(npz / "reference.npz", action=rel)
            chk("7 상대 규모(50mm) -> 통과 (정답 아는 행)",
                action_scale_report(npz)["ok"] is True,
                f"{action_scale_report(npz)['pos_abs_max_mm']:.1f} mm")
            rel[..., :3] = 0.43
            np.savez(npz / "reference.npz", action=rel)
            chk("8 절대 규모(430mm) -> 불합격 (판별행)",
                action_scale_report(npz)["ok"] is False,
                f"{action_scale_report(npz)['pos_abs_max_mm']:.1f} mm")
        except ImportError:
            chk("7 numpy 없음 -> 미판정 (통과 아님)",
                action_scale_report(root / "full")["ok"] is None)

    print(f"\n자체검증 {ok}/{tot}")
    return 0 if ok == tot else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run-dir", help="out/<NAME>/runs/<RUN_ID>")
    ap.add_argument("--dataset-report", help="<batch>.zarr.report.json")
    ap.add_argument("--exporter", default=DEF_EXPORTER)
    ap.add_argument("--umi-root", default=DEF_UMI)
    ap.add_argument("--out-root", default="~/jetson_pkg")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--verify-only", help="이미 만들어진 _jetson 폴더만 검사한다")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(selftest())

    if a.verify_only:
        out = Path(a.verify_only).expanduser()
        v = verify(out, None)
        for r in v["rows"]:
            mark = {True: " 통과 ", False: "불합격 ", None: "미판정 "}[r["ok"]]
            print(f"  [{mark}] {r['name']:<28} {r['got']}")
        print(f"\n  통과 {v['passed']} · 불합격 {v['failed']} · 미판정 {v['unknown']} "
              f"/ {v['total']}  ->  {v['verdict']}")
        raise SystemExit(0 if v["verdict"] == "PASS" else 2)

    if not (a.run_dir and a.dataset_report):
        ap.error("--run-dir 와 --dataset-report 가 필요하다 (또는 --selftest / --verify-only)")

    print("계측기 자체검증 먼저 —")
    if selftest() != 0:
        raise SystemExit("!! 자체검증 실패 — 패키지를 만들지 않는다")

    raise SystemExit(build(Path(a.run_dir).expanduser(), Path(a.dataset_report).expanduser(),
                           Path(os.path.expanduser(a.exporter)),
                           Path(os.path.expanduser(a.umi_root)),
                           Path(os.path.expanduser(a.out_root)), a.python))


if __name__ == "__main__":
    main()
