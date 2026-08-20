# 참고 — Bulk-Synchronous Parallel (BSP) 모델

> 구현 단계가 아니라 **배경 자료**다. `DOC/10`에서 "GPU가 bulk-synchronous
> 모델로 설계됐다"고 쓴 근거를 여기에 모아둔다.
>
> 링크와 출처는 문서 끝에 있다.

## 한 문장

**계산과 통신을 섞지 않고, 계산 → 통신 → 전원 동기화를 한 덩어리로 반복하는
병렬 모델.**

Leslie Valiant가 1990년 CACM 논문에서 제안했다. 그는 이 공로를 포함해 2010년
튜링상을 받았다.

---

## Superstep

BSP 프로그램은 **superstep**의 연속이다. 한 superstep 안에서:

```
[1] 지역 계산     각 프로세서가 자기 메모리만 보고 계산한다
[2] 통신          결과를 주고받는다 (아직 읽지 않는다)
[3] 배리어        전원이 도착할 때까지 기다린다
        ↓
    다음 superstep — 이제 [2]에서 보낸 것을 읽어도 안전하다
```

핵심은 **[1]과 [2]가 겹치지 않는다**는 것이다. 계산하는 동안 남이 내 메모리를
바꾸는 일이 없으므로, 락도 경합도 생각할 필요가 없다.

배리어가 superstep의 **경계**를 만든다. 배리어가 없으면 superstep이라는 개념
자체가 성립하지 않는다.

---

## 비용 모델

BSP가 단순한 프로그래밍 모델을 넘어 **성능을 예측할 수 있는** 모델인 이유가
이것이다. superstep 하나의 비용은:

```
w + h·g + l

w   그 superstep에서 가장 오래 걸린 프로세서의 지역 계산량
h   한 프로세서가 주고받은 메시지 수 중 최대 (h-relation)
g   통신 대역폭의 역수 — 기계마다 고정된 상수
l   배리어 동기화 비용 (latency) — 역시 기계 상수
```

**`g`와 `l`이 기계를 특징짓는 두 숫자다.** 알고리즘은 `w`와 `h`를 결정하고,
기계는 `g`와 `l`을 결정한다. 이 분리가 Valiant가 노린 것이다.

`w`가 **평균이 아니라 최대**인 점에 주의. 배리어가 있으므로 가장 느린
프로세서가 전체를 결정한다 — 앞서 "배리어는 병목 아닌가?"에 대한 답이 여기 있다.
병목은 배리어 자체가 아니라 **불균형**이다.

---

## 왜 "bridging model"인가

논문 제목이 *A Bridging Model for Parallel Computation*이다. 다리를 놓으려는
두 쪽은:

```
소프트웨어 (알고리즘)  ←── BSP ──→  하드웨어 (병렬 기계)
```

순차 컴퓨팅에서 **폰 노이만 모델**이 그 역할을 한다. 알고리즘을 짜는 사람은
실제 CPU가 파이프라인인지 슈퍼스칼라인지 몰라도 되고, CPU를 만드는 사람은 어떤
프로그램이 돌지 몰라도 된다. 양쪽이 같은 추상 모델에 맞추면 되니까.

Valiant는 병렬 컴퓨팅에 그런 중간층이 없다고 봤다. BSP가 그 자리를 노린 제안이다.

**"이게 최선의 병렬 모델이다"가 아니라 "양쪽이 합의할 수 있는 계약이다"** 라는
주장에 가깝다. 이 관점이 GPU를 이해하는 데도 유용하다 — CUDA의 블록/워프 구조는
성능이 최적이라서가 아니라 **하드웨어가 지킬 수 있는 약속이라서** 그 모양이다.

---

## GPU가 BSP인 지점

| BSP | CUDA / 우리 프로젝트 |
|-----|---------------------|
| 프로세서 | 스레드 |
| superstep | `__syncthreads()` 사이의 구간 |
| 지역 계산 | 레지스터 연산 |
| 통신 | 공유 메모리 읽기/쓰기 |
| 배리어 | `__syncthreads()` = 우리 `BARRIER` |
| `l` (배리어 비용) | 가장 느린 워프를 기다리는 시간 |

CUDA 문서가 `__syncthreads()`를 **"블록의 모든 스레드가 도달해야 통과하는
배리어"** 로 정의한다. BSP의 배리어와 같은 것이다.

### 어긋나는 지점도 있다

**[1] 블록 사이에는 배리어가 없다.** BSP는 전체 프로세서가 한 배리어에 모이지만,
CUDA의 배리어는 **블록 안에서만** 유효하다. 블록끼리 동기화하려면 커널을 나눠야
한다 — 우리가 패스 1과 패스 2를 따로 런치하는 것이 정확히 그것이다.

```
BSP        전역 배리어 1개
CUDA       블록 안 배리어 + 커널 경계가 전역 배리어 역할
```

**[2] 통신이 메시지가 아니라 공유 메모리다.** BSP는 원래 메시지 전달을 상정하지만,
GPU는 같은 주소 공간을 공유한다. 비용 모델의 `h`가 "메시지 수"가 아니라 "메모리
트랜잭션 수"가 되는 셈이다.

**[3] Volta 이후 워프 내부는 더 이상 lockstep이 아니다.** NVIDIA Volta가
**Independent Thread Scheduling**을 도입해 스레드마다 별도 PC와 호출 스택을
갖는다. 워프 안에서도 명시적 동기화가 필요해졌다는 뜻이다.

우리 시뮬레이터는 **자료 구조로는 Volta 쪽**이다 — `Thread::pc`가 스레드마다
있고 reconvergence 스택이 없다. 다만 스케줄링 정책이 "항상 최소 PC"라서 재수렴이
자동으로 따라오고, 그 대신 **Volta의 전진 보장은 없다**: 낮은 PC의 스레드가 높은
PC의 스레드를 기다리며 회전하면 교착한다. 자세히는 `DOC/00` §1.

---

## 다른 데서도 쓴다

BSP는 GPU 전용 개념이 아니다.

**Google Pregel** (SIGMOD 2010) — 대규모 그래프 처리 시스템. 논문이 명시적으로
"Valiant의 BSP 모델에서 영감을 받았다"고 밝힌다. 정점마다 계산 → 메시지 전송 →
superstep 경계를 반복한다. Apache Giraph, Apache Hama가 이 계열의 오픈소스
구현이다.

이게 시사하는 바: **BSP는 수천 대 클러스터에서도, 한 칩 안 256 스레드에서도
같은 모양으로 쓰인다.** 규모가 아니라 "계산과 통신을 시간축에서 분리한다"는
아이디어가 본체다.

---

## 한계

**불균형에 약하다.** 매 superstep마다 가장 느린 참여자를 기다리므로, 작업량이
고르지 않으면 그대로 손해다. 앞서 `[5c]` 설계에서 "삼각형당 스레드는 load
balance가 망가진다"고 접은 이유가 이것이다.

**세밀한 동기화를 표현하지 못한다.** 두 스레드만 짝지어 주고받고 싶어도 전원이
모여야 한다. 그래서 GPU에도 원자적 연산이 따로 있다.

**배리어가 잦으면 비용이 쌓인다.** 타일드 행렬곱이 반복마다 배리어를 두 번
치는 것이 대표적이고, double buffering이 그 완화책이다.

현대 CUDA가 **asynchronous barrier**와 **Cooperative Groups**를 추가한 것도
이 경직성에 대한 대응이다 — 전원이 멈춰 서는 대신 도착과 대기를 분리해서
통신과 계산을 겹치게 한다.

---

## 우리 프로젝트에 주는 함의

`DOC/10`에서 3안(삼각형당 스레드 + 원자적 연산)을 접고 배리어를 고른 것은
**BSP 쪽에 서겠다는 선택**이었다.

```
원자적 연산 방식   세밀한 동기화, 경합, 예측 불가능한 비용
BSP 방식           굵은 집합 지점, 경합 없음, w + h·g + l 로 예측 가능
```

그리고 우리 통계 모델이 `l`(배리어 비용)을 **0으로 본다**는 한계가 여기서
분명해진다. 발행된 일만 세므로 "가장 느린 워프를 기다린 시간"이 잡히지 않는다.
BSP 비용 모델의 세 항 중 하나가 빠져 있는 셈이고, `benchmarks/RESULTS.md`에
명시해야 할 이유다.

---

## 추가 자료

### 원전

- Leslie G. Valiant, **"A Bridging Model for Parallel Computation"**,
  *Communications of the ACM* 33(8), 1990, pp. 103–111.
  [dl.acm.org/doi/10.1145/79173.79181](https://dl.acm.org/doi/10.1145/79173.79181)
  · [dblp 항목](https://dblp.org/rec/journals/cacm/Valiant90.html)

  BSP를 처음 제안한 논문. superstep, `g`, `l`, h-relation이 전부 여기서 나온다.
  9쪽이라 읽어볼 만하다.

### BSP를 쓰는 시스템

- Malewicz et al., **"Pregel: A System for Large-Scale Graph Processing"**,
  *SIGMOD 2010*.
  [dl.acm.org/doi/10.1145/1807167.1807184](https://dl.acm.org/doi/10.1145/1807167.1807184)
  · [PDF (CMU 강의 사본)](https://15799.courses.cs.cmu.edu/fall2013/static/papers/p135-malewicz.pdf)

  BSP를 클러스터 규모 그래프 처리에 적용한 사례. "vertex-centric" 프로그래밍
  모델의 출발점이기도 하다.

### CUDA 쪽

- **CUDA Programming Guide — Writing SIMT Kernels**
  [docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)

  `__syncthreads()`의 정의와 "블록의 모든 스레드가 도달해야 한다"는 요구사항.

- **CUDA Programming Guide — Asynchronous Barriers**
  [docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)

  도착과 대기를 분리해 통신과 계산을 겹치는 최신 방식. 우리 `BARRIER`가 안 하는 것.

- **CUDA Programming Guide — Cooperative Groups**
  [docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)

  블록보다 작거나 큰 단위로 동기화하는 API. `cg::sync()`가 `__syncthreads()`와 같다.

- **Volta Tuning Guide**
  [docs.nvidia.com/cuda/volta-tuning-guide/](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html)

  Independent Thread Scheduling 도입 설명. 우리가 따르는 lockstep 가정이 어디까지
  유효한지의 경계선.

### 발산과 재수렴 (우리 스케줄러와 직접 관련)

- **"Control Flow Management in Modern GPUs"** (arXiv 2407.02944)
  [arxiv.org/pdf/2407.02944](https://arxiv.org/pdf/2407.02944)

  재수렴 방식들을 정리한 서베이. 우리가 쓰는 min-PC 방식과 실제 하드웨어의
  reconvergence stack을 비교해볼 수 있고, SIMT-induced deadlock도 다룬다.

---

[← BARRIER와 공유 메모리](./10_barrier_and_shared_memory.md) · [구현 노트 목차](./README.md)
