"""Extract the BE-contract fields that must come from a checkpoint, not from guesswork.
BE 계약에 들어갈 값 중 체크포인트에서만 나오는 것을 뽑는다.

왜 (2026-09-18)
---------------
D-AI-61 ⑤ 의 미채움 필드를 추측으로 채우지 않기로 했다. 이 도구가 재는 것:

    action horizon      정책이 한 번에 내는 스텝 수
    n_obs_steps         관측 history 길이
    obs_down_sample_steps  입력 레이트 (실 10Hz→1, 시뮬 30Hz→3)
    learningRate        공식 UMI 기본값을 쓰고 있어 우리가 명시한 적이 없다
    nParams             미측정
    action dim          shape_meta 의 action 차원

⚠️ 2026-09-18 실증 — 어시스턴트가 `horizon: 8` 을 확인 없이 BE 에 넘길 뻔했다.
   v10 청크가 8인 것과 학습 action horizon 은 다른 값이다.

계측기 규칙
-----------
- **모수를 같이 찍는다.** `찾음 N / 전체 M` 형태로. ckpt 탐색·키 탐색 전부
- **검색 0건이면 범위가 완전한지 먼저 본다.** 0건과 "탐색 실패"를 구분해 출력한다
- **가용성 확인 ≠ 기능 확인.** cfg 만 읽지 않고 state_dict 를 실제로 순회해 센다
- **정답을 아는 행을 넣는다.** 파라미터 수를 아는 가짜 state_dict 로 계수기를 검산한다

Usage
-----
  python probe_ckpt_contract.py --selftest
  python probe_ckpt_contract.py --ckpt <경로.ckpt>
  python probe_ckpt_contract.py --root ~/handoff        # 하위 *.ckpt 전수
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 찾고 싶은 키. 경로를 가정하지 않고 재귀 탐색한다 — 설정 구조가 바뀌어도 안 깨진다.
WANTED = ("horizon", "n_action_steps", "n_obs_steps", "obs_down_sample_steps",
          "lr", "learning_rate", "num_inference_steps", "shape_meta")


# ── 순수 계산 — 자체검증 대상 ─────────────────────────────────────────────

def count_params(sd: dict) -> tuple[int, int, int]:
    """Sum numel over tensor entries. 텐서 항목의 원소 수 합계.
    반환 (파라미터 수, 텐서 수, 전체 항목 수) — 모수를 같이 낸다."""
    import torch
    n = t = 0
    for v in sd.values():
        if torch.is_tensor(v):
            n += v.numel(); t += 1
    return (n, t, len(sd))


def find_keys(node: Any, wanted: tuple[str, ...], path: str = "",
              out: dict[str, Any] | None = None, depth: int = 0) -> dict[str, Any]:
    """Recursively collect wanted keys with their full paths.
    원하는 키를 전체 경로와 함께 재귀 수집한다. 경로를 가정하지 않는다."""
    if out is None:
        out = {}
    if depth > 12:
        return out
    items: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        items = list(node.items())
    elif hasattr(node, "items") and not isinstance(node, (str, bytes)):
        try:
            items = list(node.items())                                 # OmegaConf
        except Exception:                                              # noqa: BLE001
            items = []
    for k, v in items:
        p = f"{path}.{k}" if path else str(k)
        if str(k) in wanted and not isinstance(v, (dict,)) or str(k) == "shape_meta":
            out[p] = _plain(v)
        find_keys(v, wanted, p, out, depth + 1)
    return out


def _plain(v: Any) -> Any:
    """Make a config value JSON-safe. 설정값을 JSON 으로 낼 수 있게."""
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    try:
        return json.loads(json.dumps(v, default=str))
    except Exception:                                                  # noqa: BLE001
        return str(v)


# ── 자체검증 ─────────────────────────────────────────────────────────────

def selftest() -> int:
    import torch
    bad = 0

    sd = {"a": torch.zeros(3, 4), "b": torch.zeros(10), "note": "텐서 아님"}
    n, t, tot = count_params(sd)
    ok = n == 22 and t == 2 and tot == 3
    print(f"[1] 파라미터 계수 → {n} (텐서 {t}/{tot})  기대 22 (2/3)  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    n, t, tot = count_params({"only": "문자열"})
    ok = n == 0 and t == 0 and tot == 1
    print(f"[2] 텐서 0개 → {n} (텐서 {t}/{tot})  기대 0 (0/1)  ", end="")
    print("OK" if ok else "!! 실패 — 비텐서를 세고 있다"); bad += (not ok)

    n, t, tot = count_params({})
    ok = n == 0 and t == 0 and tot == 0
    print(f"[3] 빈 state_dict → {n} (텐서 {t}/{tot})  기대 0 (0/0)  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    cfg = {"policy": {"horizon": 16, "n_action_steps": 8,
                      "noise_scheduler": {"num_inference_steps": 16}},
           "optimizer": {"lr": 3e-4}, "task": {"obs_down_sample_steps": 3}}
    f = find_keys(cfg, WANTED)
    ok = (f.get("policy.horizon") == 16 and f.get("optimizer.lr") == 3e-4
          and f.get("task.obs_down_sample_steps") == 3
          and f.get("policy.noise_scheduler.num_inference_steps") == 16)
    print(f"[4] 중첩 키 전체경로 수집 → {len(f)}개  ", end="")
    print("OK" if ok else f"!! 실패 — {f}"); bad += (not ok)

    f = find_keys({"a": {"b": {"c": 1}}}, WANTED)
    ok = len(f) == 0
    print(f"[5] 없는 키는 0건 → {len(f)}개  기대 0  ", end="")
    print("OK" if ok else "!! 실패 — 없는 걸 만들어낸다"); bad += (not ok)

    # 고의 오답: horizon 이 두 군데 있으면 둘 다 나와야 한다 (하나만 집으면 조용히 틀린다)
    f = find_keys({"policy": {"horizon": 16}, "task": {"horizon": 8}}, WANTED)
    ok = len(f) == 2 and f.get("policy.horizon") == 16 and f.get("task.horizon") == 8
    print(f"[6] horizon 중복 정의 둘 다 포착 → {len(f)}개  기대 2  ", end="")
    print("OK" if ok else f"!! 실패 — {f}"); bad += (not ok)

    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


# ── 체크포인트 판독 ──────────────────────────────────────────────────────

def probe(p: Path) -> dict:
    import torch
    r: dict[str, Any] = {"ckpt": str(p), "bytes": p.stat().st_size}
    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:                                           # noqa: BLE001
        r["error"] = f"{type(exc).__name__}: {exc}"
        return r

    r["top_keys"] = sorted(str(k) for k in payload) if isinstance(payload, dict) else [type(payload).__name__]

    cfg = payload.get("cfg") if isinstance(payload, dict) else None
    if cfg is None:
        r["cfg"] = "없음 — payload 에 cfg 키가 없다"
    else:
        found = find_keys(cfg, WANTED)
        r["cfg_found"] = found
        r["cfg_found_n"] = len(found)
        if not found:
            r["cfg_note"] = "⚠️ 0건. 키 이름이 바뀌었을 수 있다. top_keys 를 보고 범위를 확인하라"

    sds = payload.get("state_dicts") if isinstance(payload, dict) else None
    if not isinstance(sds, dict):
        sds = {k: v for k, v in payload.items()
               if isinstance(v, dict) and any(hasattr(x, "numel") for x in v.values())} \
            if isinstance(payload, dict) else {}
    r["state_dict_names"] = sorted(str(k) for k in sds)
    r["params"] = {}
    for name, sd in sds.items():
        if isinstance(sd, dict):
            n, t, tot = count_params(sd)
            r["params"][str(name)] = {"nParams": n, "tensors": t, "entries": tot}
    if not r["params"]:
        r["params_note"] = "⚠️ state_dict 를 못 찾았다. 0 이 아니라 판정 불가다"
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--root")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 판정하지 않는다")
    print()

    paths = [Path(c).expanduser() for c in a.ckpt]
    scanned = 0
    if a.root:
        root = Path(a.root).expanduser()
        found = sorted(root.rglob("*.ckpt")) + sorted(root.rglob("*.pt"))
        scanned = len(found)
        paths += found
    paths = [p for p in dict.fromkeys(paths) if p.exists()]
    print(f"대상 {len(paths)}개" + (f" / 탐색 {scanned}개" if a.root else "")
          + (f"  (루트 {a.root})" if a.root else ""))
    if not paths:
        raise SystemExit("!! 체크포인트 0개. **없는 게 아니라 못 찾은 것일 수 있다** — "
                         "--root 경로가 맞는지, 확장자가 .ckpt/.pt 인지 먼저 확인하라")

    rows = []
    for i, p in enumerate(paths, 1):
        print(f"\n[{i}/{len(paths)}] {p}")
        r = probe(p)
        rows.append(r)
        if "error" in r:
            print(f"  !! 로드 실패 {r['error']}"); continue
        print(f"  크기        {r['bytes']/1e6:.1f} MB")
        print(f"  top keys    {r['top_keys']}")
        for k, v in sorted(r.get("cfg_found", {}).items()):
            if k.endswith("shape_meta"):
                continue
            print(f"  cfg  {k:48s} = {v}")
        sm = {k: v for k, v in r.get("cfg_found", {}).items() if k.endswith("shape_meta")}
        for k, v in sm.items():
            print(f"  cfg  {k} =")
            print(f"       {json.dumps(v, ensure_ascii=False)[:600]}")
        if "cfg_note" in r:
            print(f"  {r['cfg_note']}")
        for name, d in r["params"].items():
            print(f"  params  {name:20s} nParams {d['nParams']:,}  "
                  f"(텐서 {d['tensors']}/{d['entries']})")
        if "params_note" in r:
            print(f"  {r['params_note']}")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
