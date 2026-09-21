#!/usr/bin/env python3
"""Capture gate as an HTTP service — judge a recording BEFORE the video is uploaded.
촬영 게이트를 HTTP 로 낸다. **영상을 올리기 전에** 판정한다.

왜 서비스인가
-------------
촬영 게이트 15항목은 전부 **메타데이터만으로** 나온다 (manifest · frames.csv · IMU csv).
영상 16MB 를 올릴 필요가 없다. 수백 KB 면 1초 안에 답이 나오고, **사람이 아직 현장에
있을 때** "다시 찍으세요"가 뜬다. SLAM 세 시간 돌리고 알면 늦다.

왜 앱이 아니라 여기인가
-----------------------
게이트 로직을 앱(Kotlin)에 다시 구현하면 파이썬 판정과 갈린다. 그 순간 같은 데이터에
두 개의 답이 생기고 어느 쪽이 맞는지 아무도 모른다. **구현은 하나다.** 앱은 올리고
결과를 보여주기만 한다.

계약 (BE 최은찬 합의 필요)
---------------------------
POST /gate/capture   Content-Type: application/json
{
  "episode": "rec_1789701868830_2c55035d",
  "video_bytes": 16756474,          // 영상 자체는 안 보낸다. 크기만. 없으면 **미판정**
  "files": {
     "manifest.json":     "<파일 내용 그대로>",
     "frames.csv":        "...",
     "accelerometer.csv": "...",
     "gyroscope.csv":     "...",
     "encoded.csv":       "..."     // 선택
  }
}
->  200 {"episode":..., "verdict":"PASS|FAIL|INCOMPLETE",
         "passed":n,"failed":n,"unknown":n,"total":n,
         "soft_unmet":n,"soft_total":n,
         "rows":[{"name","ok","got","want","soft"}],
         "measured":{...}, "gate_version":"..."}
    400 {"error": "...", "missing": [...]}          // 필수 파일 누락 등

GET /gate/spec     게이트 기준값 전부 (앱이 사용자에게 "왜 떨어졌나"를 설명할 때 쓴다)
GET /health        살아있음 + 기준값 해시. **가용성 확인이지 기능 확인이 아니다**

⚠️ 미판정(INCOMPLETE)은 통과가 아니다. `verdict != "PASS"` 면 올리지 않는다.

Usage
    python gate_api.py --selftest
    python gate_api.py --serve --port 8971
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from gate_raw_capture import (GATES, REQUIRED_ORIENT, REQUIRED_OUTCOME,  # noqa: E402
                              REQUIRED_STATUS, judge, measure_episode)

GATE_VERSION = "capture-gate-v2"          # 필수/권고 분리판 (2026-09-21)
REQUIRED_FILES = ("manifest.json", "frames.csv")
OPTIONAL_FILES = ("accelerometer.csv", "gyroscope.csv", "encoded.csv")
MAX_BODY = 32 << 20                        # 32MB. 메타데이터만 받으므로 넉넉하다


def spec() -> dict:
    """The thresholds this service enforces. 이 서비스가 거는 기준 전부."""
    body = {"gate_version": GATE_VERSION, "gates": GATES,
            "required_outcome": REQUIRED_OUTCOME, "required_status": REQUIRED_STATUS,
            "required_orientation": REQUIRED_ORIENT,
            "required_files": list(REQUIRED_FILES), "optional_files": list(OPTIONAL_FILES)}
    body["spec_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return body


def evaluate(payload: dict) -> tuple[int, dict]:
    """Judge one capture from its metadata. 메타데이터만으로 한 편을 판정한다.
    반환 (HTTP 상태, 본문)."""
    if not isinstance(payload, dict):
        return 400, {"error": "본문이 객체가 아니다"}
    files = payload.get("files")
    if not isinstance(files, dict):
        return 400, {"error": "files 가 없다", "missing": list(REQUIRED_FILES)}
    missing = [f for f in REQUIRED_FILES if not isinstance(files.get(f), str)]
    if missing:
        return 400, {"error": "필수 파일이 없다", "missing": missing,
                     "received": sorted(files)}

    name = str(payload.get("episode") or "rec_unnamed")
    if "/" in name or "\\" in name or name.startswith("."):
        return 400, {"error": f"episode 이름이 경로를 포함한다: {name!r}"}

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / name
        d.mkdir(parents=True)
        for fn, text in files.items():
            if fn in REQUIRED_FILES + OPTIONAL_FILES and isinstance(text, str):
                (d / fn).write_text(text, encoding="utf-8")
        try:
            r = measure_episode(d)
        except Exception as exc:                        # noqa: BLE001
            return 400, {"error": f"측정 실패: {type(exc).__name__}: {exc}"}

    # 영상은 안 받는다. 크기를 주면 그 항목을 판정하고, 안 주면 **미판정**으로 남긴다.
    vb = payload.get("video_bytes")
    if isinstance(vb, (int, float)) and vb > 0:
        r["video_bytes"] = int(vb)
    elif vb is None and "video_bytes" not in payload:
        r.pop("video_bytes", None)          # 안 알려줬다 -> **미판정**. 불합격이 아니다
    else:
        r["video_bytes"] = None             # 0 이나 잘못된 값 -> 없음으로 본다
    j = judge(r)

    body = {"episode": name, "gate_version": GATE_VERSION,
            "verdict": j["verdict"], "passed": j["passed"], "failed": j["failed"],
            "unknown": j["unknown"], "total": j["total"],
            "soft_unmet": j["soft_unmet"], "soft_total": j["soft_total"],
            "rows": j["rows"], "measured": {k: v for k, v in r.items() if k != "gates"},
            "note": "미판정은 통과가 아니다. verdict != PASS 면 영상을 올리지 않는다"}
    return 200, body


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):                  # 표준 액세스 로그는 시끄럽다
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False, indent=1).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):                                    # noqa: N802
        if self.path.rstrip("/") == "/health":
            s = spec()
            self._send(200, {"ok": True, "gate_version": GATE_VERSION,
                             "spec_sha256": s["spec_sha256"],
                             "note": "살아있음 확인이다. 기능 확인은 /gate/capture 로 해라"})
        elif self.path.rstrip("/") == "/gate/spec":
            self._send(200, spec())
        else:
            self._send(404, {"error": f"없는 경로: {self.path}"})

    def do_POST(self):                                   # noqa: N802
        if self.path.rstrip("/") != "/gate/capture":
            self._send(404, {"error": f"없는 경로: {self.path}"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send(400, {"error": "Content-Length 가 숫자가 아니다"})
            return
        if n <= 0 or n > MAX_BODY:
            self._send(400, {"error": f"본문 길이 {n} — 1~{MAX_BODY} 바이트여야 한다"})
            return
        raw = self.rfile.read(n)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:                         # noqa: BLE001
            self._send(400, {"error": f"JSON 파싱 실패: {exc}"})
            return
        code, body = evaluate(payload)
        self._send(code, body)


def serve(port: int, host: str = "0.0.0.0") -> None:
    srv = ThreadingHTTPServer((host, port), Handler)
    s = spec()
    print(f"촬영 게이트 서비스 — http://{host}:{port}")
    print(f"  gate_version {GATE_VERSION} · spec_sha256 {s['spec_sha256'][:16]}")
    print(f"  필수 {len([1 for _ in GATES])}개 기준 · POST /gate/capture · GET /gate/spec · GET /health")
    srv.serve_forever()


# ── 자체검증 ────────────────────────────────────────────────────────────────
def _payload_from_dir(d: Path, *, video_bytes: int | None = 16756474) -> dict:
    files = {}
    for fn in REQUIRED_FILES + OPTIONAL_FILES:
        p = d / fn
        if p.exists():
            files[fn] = p.read_text(encoding="utf-8")
    body = {"episode": d.name, "files": files}
    if video_bytes is not None:
        body["video_bytes"] = video_bytes
    return body


def selftest() -> int:
    import urllib.request
    from gate_raw_capture import _write_synth

    ok = tot = 0

    def chk(name, cond, note=""):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(f"  {'OK  ' if cond else '실패'} {name}   {note}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        good = _write_synth(root, "rec_good", frames=100)
        short = _write_synth(root, "rec_short", frames=40)
        long_ = _write_synth(root, "rec_long", frames=250)

        # [1-4] evaluate() 직접 — 서버 없이 판정 로직
        c, b = evaluate(_payload_from_dir(good))
        chk("1 정상 -> 200 PASS", c == 200 and b["verdict"] == "PASS",
            f"{c} · {b.get('verdict')} · 필수 {b.get('passed')}/{b.get('total')}")
        c2, b2 = evaluate(_payload_from_dir(short))
        chk("2 40프레임 -> FAIL (판별행)", c2 == 200 and b2["verdict"] == "FAIL",
            f"{b2.get('verdict')}")
        c3, b3 = evaluate(_payload_from_dir(long_))
        chk("3 250프레임 -> PASS + 권고 미충족 1 (정답 아는 행)",
            b3["verdict"] == "PASS" and b3["soft_unmet"] == 1,
            f"{b3.get('verdict')} · 권고 {b3.get('soft_unmet')}/{b3.get('soft_total')}")
        c4, b4 = evaluate(_payload_from_dir(good, video_bytes=None))
        chk("4 video_bytes 없음 -> 미판정, PASS 아님 (판별행)",
            b4["verdict"] != "PASS" and b4["unknown"] >= 1,
            f"{b4.get('verdict')} · 미판정 {b4.get('unknown')}")

        # [5-7] 잘못된 입력
        c5, b5 = evaluate({"episode": "x", "files": {"frames.csv": "a"}})
        chk("5 필수 파일 누락 -> 400 + 누락 목록", c5 == 400 and b5.get("missing"),
            f"{c5} · {b5.get('missing')}")
        c6, _ = evaluate({"episode": "../etc", "files":
                          {f: "x" for f in REQUIRED_FILES}})
        chk("6 경로 포함 이름 거부 (판별행)", c6 == 400, str(c6))
        c7, _ = evaluate("문자열")
        chk("7 객체 아님 -> 400", c7 == 400, str(c7))

        # [8] 결정성 — 같은 입력 두 번이면 같은 판정
        _, b8a = evaluate(_payload_from_dir(good))
        _, b8b = evaluate(_payload_from_dir(good))
        chk("8 같은 입력 -> 같은 판정", b8a["rows"] == b8b["rows"], "행 전부 동일")

        # [9-12] 실제 HTTP 로 띄워서 — **가용성 확인이 아니라 기능 확인**
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}"
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                h = json.loads(r.read())
            chk("9 GET /health", h.get("ok") is True, h.get("gate_version"))
            with urllib.request.urlopen(base + "/gate/spec", timeout=5) as r:
                sp = json.loads(r.read())
            chk("10 GET /gate/spec 기준값 동봉",
                sp.get("spec_sha256") == spec()["spec_sha256"] and "frames_max" in sp["gates"],
                f"기준 {len(sp['gates'])}개")
            req = urllib.request.Request(
                base + "/gate/capture",
                data=json.dumps(_payload_from_dir(good)).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                hb = json.loads(r.read())
            chk("11 POST /gate/capture -> PASS (실제 HTTP)",
                hb["verdict"] == "PASS" and hb["total"] == b["total"],
                f"{hb['verdict']} · 필수 {hb['passed']}/{hb['total']}")
            bad = urllib.request.Request(base + "/gate/capture", data=b"{not json",
                                         headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(bad, timeout=5)
                code = 200
            except urllib.error.HTTPError as e:
                code = e.code
            chk("12 깨진 JSON -> 400 (판별행)", code == 400, str(code))
        finally:
            srv.shutdown()

    print(f"\n자체검증 {ok}/{tot}")
    return 0 if ok == tot else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8971)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if not a.serve:
        ap.error("--serve 또는 --selftest 가 필요하다")
    print("계측기 자체검증 먼저 —")
    if selftest() != 0:
        raise SystemExit("!! 자체검증 실패 — 서비스를 올리지 않는다")
    print()
    serve(a.port, a.host)


if __name__ == "__main__":
    main()
