# TS — V100 에서 학습이 "CUDA 커널 없음" 으로 죽었다

2026-09-16 · 김준태(트랙 B) · 서버 Tesla V100-PCIE-32GB · TLJH · sudo 없음

**이 문서는 "고쳤다" 가 아니라 "어떻게 좁혔나" 를 적는다.**

---

## 증상

서버에서 학습을 띄우면 즉시 죽었다.

```
CUDA error: no kernel image is available for execution on the device
```

그런데 **사전 점검은 전부 통과했다.**

```python
torch.cuda.is_available()          # True
torch.cuda.get_device_capability() # (7, 0)
torch.cuda.get_device_name(0)      # Tesla V100-PCIE-32GB
```

## 왜 안 좁혀졌나 — 첫 번째 막다른 길

"GPU 를 못 본다" 가 아니라 "GPU 를 보는데 못 쓴다" 라서
드라이버·`CUDA_VISIBLE_DEVICES`·권한 쪽을 먼저 의심했다. 전부 정상이었다.

이 단계에서 시간을 쓴 이유는 명확하다 —
**통과한 점검을 근거로 그 방향을 배제해버렸기 때문이다.**

## 좁힌 한 수

`is_available()` 과 `get_device_capability()` 는 **드라이버와 장치**를 묻는다.
**그 파이썬 휠에 이 장치용 커널이 들어 있는지**는 묻지 않는다. 그건 다른 질문이고,
다른 API 로 물어야 한다.

```python
torch.cuda.get_arch_list()
```

```
torch 2.11.0+cu128  →  sm_75 이상만. sm_70 없음
```

V100 은 Compute Capability 7.0 = sm_70 이다. **휠에 커널이 없다.**
드라이버도 장치도 정상이고, 빌드에 이 세대가 빠져 있었을 뿐이다.

## 확인 — 존재가 아니라 동작을 시킨다

`get_arch_list()` 에 `sm_70` 이 있다고 끝이 아니다. 실제로 연산을 시켰다.

```python
import torch
assert 'sm_70' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
print((torch.ones(4, device='cuda') * 1).tolist())
```

```
torch 2.13.0+cu126
['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']
실연산: [1.0, 1.0, 1.0, 1.0]
```

이후 학습이 끝까지 돌았다 — `Official training finished` 🟢.

## 남은 부작용

`--force-reinstall` 이 `numpy 2.2.6 → 2.5.2` 도 같이 올렸다.
opencv / numba / zarr 와의 충돌 여부는 **미확인**이다.
학습·생성·export·평가가 전부 돌았으므로 치명적이진 않지만,
BE 에 넘긴 `requirements-sim-server.txt` 에 이 사실을 명시했다.

## 일반화 — 규칙 ⑩ 으로 박았다

**가용성 확인은 기능 확인이 아니다.**

같은 모양이 여럿이다.

| 존재·가용 검사 | 안 보는 것 |
|---|---|
| `torch.cuda.is_available()` | 이 빌드에 이 아키텍처 커널이 있는지 |
| `which ffmpeg` | 실행 권한 |
| `import cv2` 성공 | 필요한 코덱 유무 |
| 파일 존재 | 읽기 권한 |

**"쓸 수 있는가" 를 묻지 말고 "실제로 되는가" 를 시켜라.**

## 파생 — 로컬은 반대 방향으로 갈린다

같은 함정이 로컬에서 부호만 바꿔 나온다.

```
V100        sm_70   → cu126 필요.  cu128 빌드는 sm_75+ 전용이라 못 씀
RTX 5070    sm_120  → cu128 이상 필요. cu126 으로는 못 씀
RTX 4070    sm_89   → cu126 가능 (같은 major 의 sm_86 cubin 이 실행됨) 🔵
```

**서버와 로컬은 torch 빌드가 달라야 한다.** 하나로 통일하려 들면 한쪽이 죽는다.
