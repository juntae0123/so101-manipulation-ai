"""Write a smaller official-UMI zarr containing the first N episodes.
공식 UMI zarr 에서 앞 N 편만 잘라 작은 zarr 로 다시 쓴다.

왜 (2026-09-18)
---------------
`v4.zarr` 가 1.5GB 인데 Jupyter 브라우저 업로드가 조각마다 잘렸다
(400MB 조각 4개가 72/67/64/86MB 로 도착). 전체를 올리는 대신
**앞 N 편만 올려 파이프라인을 먼저 검증한다** — 단계식 수집 원칙과 같다.

⚠️ 이건 실험용 부분집합이다. **여기서 나온 성공률을 전체 데이터 성능으로 인용하지 마라.**
   편수가 줄면 그 자체로 성공률이 달라진다.

⚠️ `episode_ends` 는 누적합이다. 앞 N 편을 자르면 각 배열을 `episode_ends[N-1]` 까지
   자르고 `episode_ends` 도 앞 N 개만 남긴다. **행 수와 경계가 어긋나면 학습이
   조용히 엉뚱한 구간을 섞는다.** 그래서 쓰고 나서 다시 읽어 검산한다.

Usage
-----
  python subset_umi_zarr.py --selftest
  python subset_umi_zarr.py --src v4.zarr --dst v4_head20.zarr --episodes 20
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


# ── 순수 계산 — 자체검증 대상 ─────────────────────────────────────────────

def cut_point(episode_ends: np.ndarray, n: int) -> tuple[int, np.ndarray]:
    """Row count and new episode_ends for the first n episodes.
    앞 n 편의 행 수와 새 episode_ends. n 이 범위를 넘으면 전체를 돌려준다."""
    if len(episode_ends) == 0:
        return (0, episode_ends[:0])
    n = max(1, min(n, len(episode_ends)))
    return (int(episode_ends[n - 1]), episode_ends[:n].copy())


def selftest() -> int:
    bad = 0
    ee = np.array([10, 25, 40, 60], dtype=np.int64)

    rows, new = cut_point(ee, 2)
    ok = rows == 25 and list(new) == [10, 25]
    print(f"[1] 앞 2편 → 행 {rows} · 경계 {list(new)}  기대 25 / [10, 25]  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    rows, new = cut_point(ee, 99)
    ok = rows == 60 and len(new) == 4
    print(f"[2] 범위 초과 → 행 {rows} · 편 {len(new)}  기대 60 / 4  ", end="")
    print("OK" if ok else "!! 실패 — 넘치면 잘라야 한다"); bad += (not ok)

    rows, new = cut_point(ee, 0)
    ok = rows == 10 and len(new) == 1
    print(f"[3] 0 요청 → 행 {rows} · 편 {len(new)}  기대 10 / 1 (0편은 무의미)  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    rows, new = cut_point(np.array([], dtype=np.int64), 5)
    ok = rows == 0 and len(new) == 0
    print(f"[4] 빈 입력 → 행 {rows} · 편 {len(new)}  기대 0 / 0  ", end="")
    print("OK" if ok else "!! 실패"); bad += (not ok)

    # 정답을 아는 오답: 마지막 경계가 행 수와 다르면 잘못 자른 것이다
    rows, new = cut_point(ee, 3)
    ok = rows == int(new[-1])
    print(f"[5] 마지막 경계 == 행 수 → {int(new[-1])} vs {rows}  ", end="")
    print("OK" if ok else "!! 실패 — 경계와 행 수가 어긋난다"); bad += (not ok)

    # [6] 실제 .zarr.zip 왕복 — 이게 없어서 "zip 을 안 닫는다"를 못 잡았다.
    #     순수 계산만 검증하면 저장 경로의 결함은 통과한다.
    try:
        import tempfile, zarr, os
        with tempfile.TemporaryDirectory() as td:
            dst = os.path.join(td, "t.zarr.zip")
            g, store = _open_dst(dst)
            try:
                g.create_group("data").create_dataset("x", data=np.arange(6))
                g.create_group("meta").create_dataset(
                    "episode_ends", data=np.array([3, 6], dtype=np.int64))
            finally:
                if store is not None:
                    store.close()
            z = zarr.open(dst, mode="r")          # 닫힌 뒤 **새로** 연다
            ok = (list(z["data"]["x"][:]) == [0, 1, 2, 3, 4, 5]
                  and list(z["meta"]["episode_ends"][:]) == [3, 6])
        print(f"[6] .zarr.zip 쓰고 새로 읽기  ", end="")
        print("OK" if ok else "!! 실패 — zip 이 확정되지 않았다"); bad += (not ok)
    except ImportError:
        print("[6] .zarr.zip 왕복  **미실행** (zarr 없음). 통과로 세지 않는다"); bad += 1

    print(f"\n자체검증 {'통과' if bad == 0 else f'실패 {bad}건'}")
    return 1 if bad else 0


def _open_dst(path: str):
    """Open the destination, returning (group, store_to_close).
    출력 대상을 연다. zip 이면 **닫아야 할 store 를 같이** 돌려준다.

    ⚠️ 2026-09-19: `zarr.open(path, mode="w")` 로 `.zarr.zip` 을 쓰면 store 가
    닫히지 않아 중앙 디렉터리가 안 써진다. 되읽을 때 `BadZipFile` 이 난다.
    디렉터리 zarr 로만 써봐서 여기까지 안 왔었다."""
    import zarr
    if str(path).endswith(".zip"):
        store = zarr.ZipStore(path, mode="w")
        # overwrite=True 는 rmdir 을 부르는데 ZipStore 가 제대로 못 한다. 새 파일이라 불필요
        return zarr.group(store=store), store
    return zarr.open(path, mode="w"), None


# ── 실행 ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    print("계측기 자체검증 먼저 —")
    if selftest():
        raise SystemExit("!! 자체검증 실패. 실행하지 않는다")
    print()
    if not (a.src and a.dst):
        raise SystemExit("!! --src 와 --dst 가 필요하다")

    import zarr
    src = zarr.open(a.src, mode="r")
    ee = np.asarray(src["meta"]["episode_ends"][:])
    rows, new_ee = cut_point(ee, a.episodes)
    print(f"원본 편 {len(ee)} · 총 행 {int(ee[-1]) if len(ee) else 0}")
    print(f"자를 편 {len(new_ee)} · 행 {rows}  ({rows/int(ee[-1])*100:.1f}%)" if len(ee) else "")

    dst, store = _open_dst(a.dst)
    try:
        dg, sg = dst.create_group("data"), src["data"]
        keys = sorted(sg.array_keys())
        print(f"\n배열 {len(keys)}개 복사")
        for k in keys:
            arr = sg[k]
            out = np.asarray(arr[:rows])
            dg.create_dataset(k, data=out, chunks=arr.chunks, dtype=arr.dtype)
            print(f"  {k:32s} {arr.shape} → {out.shape}")
        mg = dst.create_group("meta")
        mg.create_dataset("episode_ends", data=new_ee, dtype=new_ee.dtype)
    finally:
        if store is not None:
            store.close()          # ← 닫아야 zip 이 확정된다
            print("\nZipStore 닫음")

    # 검산 — **닫은 뒤 새로 연다.** 같은 핸들로 확인하면 확정 여부를 못 본다
    import zarr
    chk = zarr.open(a.dst, mode="r")
    cee = np.asarray(chk["meta"]["episode_ends"][:])
    shapes = {k: chk["data"][k].shape[0] for k in sorted(chk["data"].array_keys())}
    uniq = sorted(set(shapes.values()))
    ok = (len(cee) == len(new_ee)) and (len(uniq) == 1) and (uniq[0] == int(cee[-1]))
    print(f"\n되읽기 검산  편 {len(cee)}/{len(new_ee)} · 배열 {len(shapes)}개 행 {uniq} · "
          f"마지막경계 {int(cee[-1])}  {'OK' if ok else '!! 경계와 행 수가 어긋난다'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
