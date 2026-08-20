# 11 — 실제 하드웨어와의 간극

> 구현 단계가 아니라 **자기 점검 문서**다. 지금까지 런타임(SIMT 실행 모델)에
> 집중하느라 그래픽스 하드웨어 쪽을 얼마나 비워뒀는지 정리하고, 각 구멍을
> 메우는 데 무엇이 드는지 값을 매긴다.
>
> README의 "What this models, and what it does not" 절이 이 문서의 요약이다.
> 여기에는 **왜 그렇게 뒀는지**와 **메우려면 무엇을 고쳐야 하는지**를 남긴다.
>
> **기계 자체의 정의는 [00_machine_spec.md](00_machine_spec.md) 에 있다.** 이
> 문서는 그것과 실제 하드웨어의 차이를 다룬다 — 명세에 "없다"고 적힌 것마다
> 여기에 항목이 하나씩 있다.

## 한 문장

**우리가 만든 것은 고정 기능 그래픽스 파이프라인의 모형이 아니라,
시뮬레이션된 SIMT 머신 위에서 도는 compute 기반 소프트웨어 렌더러다.**

래스터화가 고정 기능 블록이 아니라 커널이다. `build_raster_program`은
`build_raytrace_program`과 같은 25개 opcode로 쓰였고, 특별 대접을 받지 않는다.

### 이게 실수가 아닌 이유

고정 기능 래스터라이저를 시뮬레이션했다면 **divergence를 잴 것이 없었다.**
하드웨어 래스터라이저는 워프라는 개념 자체가 없는 곳이다. 이 프로젝트의 중심
질문("레인이 갈라지면 얼마를 무는가")은 래스터화를 커널로 했기 때문에 물을 수
있는 것이다.

같은 계열의 실제 사례:

- **Larrabee** (Intel, 2008) — 고정 기능 래스터라이저를 빼고 x86 코어 배열
  위에서 소프트웨어로 래스터화하려던 시도
- **CUDA 소프트웨어 래스터화** — Laine & Karras, *High-Performance Software
  Rasterization on GPUs*, HPG 2011
- **Nanite** (Unreal Engine 5) — 화면에서 작은 삼각형은 고정 기능
  래스터라이저를 쓰지 않고 compute 셰이더로 래스터화한다. 픽셀보다 작은
  삼각형에서는 고정 기능 쪽이 오히려 느리기 때문

**문제는 위치가 아니라 그 위치를 명시하지 않은 것이었다.** 읽는 사람이
"고정 기능을 몰라서 안 만든 것"인지 "일부러 compute로 간 것"인지 구분할
수 없다. README 절이 그것을 메운다.

---

## 대조표

### 그래픽스 파이프라인

| 하드웨어 블록 | 성격 | 우리 | 메모 |
|---|---|---|---|
| Input Assembler + post-transform vertex cache | 고정 | **없음** | `[9]`에서 다룬다 |
| Vertex Shader | 프로그래머블 | ✅ `build_vertex_program` | 정점당 스레드 하나 |
| Primitive Assembly / Clip / Cull | 고정 | 부분 | 뒷면 컬링만, 커널 안. **클리핑 없음** |
| Rasteriser (edge function, quad) | **고정** | 커널로 대체 | 이 프로젝트의 성격을 정하는 지점 |
| Hi-Z / Early-Z | 고정 | **없음** | |
| Fragment Shader | 프로그래머블 | 래스터 커널 안에 섞임 | 분리 안 함 |
| ROP (blend, depth write) | 고정 | **없음** | nearest-wins를 레지스터에서 |
| Texture Unit / Sampler | 고정 | **없음** | |

### 레이 트레이싱

| 하드웨어 | 우리 | 비고 |
|---|---|---|
| **BVH / 가속 구조 (BLAS·TLAS)** | **없음** | 픽셀마다 모든 삼각형을 선형 순회 |
| 순회 유닛 (RT 코어) | **없음** | 교차 판정을 커널이 한다 — 래스터화와 같은 선택 |
| 인덱스 버퍼 (BLAS 빌드 입력) | **없음** | 아래 참고 |
| 인스턴스 변환 (TLAS) | **없음** | |

### 메모리

| | 우리 | 비고 |
|---|---|---|
| L1 / L2 / VRAM 계층 | **없음** | 글로벌 로드 비용 **100 고정** |
| Coalescing | **없음** | 주소 패턴 무관, 레인당 과금 |
| Shared memory | ✅ | 블록당 4096 float, 로드 비용 8 |
| `BARRIER` | ✅ | → `DOC/10` |
| 레지스터 파일 | ✅ | 스레드당 256, 초과 시 예외 |

### 스케줄링

| | 우리 | |
|---|---|---|
| 워프 32레인, `activeMask` | ✅ | 프로젝트의 중심 |
| 스레드별 PC | ✅ | Volta 2017 쪽 → `DOC/00` §1 |
| 스택 없는 재수렴 | ✅ | min-PC → `DOC/04` |
| **전진 보장 (ITS)** | **없음** | `[m1]`. 자료 구조는 이미 있다 |
| **warp-level primitives** | **없음** | `[m2]`. shuffle, ballot, syncwarp |
| SM 여러 개 / occupancy | **없음** | `[c]`. 블록 순차 실행 |
| Latency hiding | **없음** | `[c]` |
| **비동기 런치 / 스트림** | **없음** | `[m4]` |
| **비동기 복사 (`cp.async`)** | **없음** | `[m5]` |
| **텐서 / 행렬 유닛** | **없음** | `[m6]` |

---

## 구멍별 값 매기기

### 1. Coalescing과 메모리 계층 — **가장 크다**

**왜 중요한가.** 실제 GPU 커널을 튜닝할 때 첫 번째 질문이다. 인접한 32레인이
인접한 32 float을 읽으면 트랜잭션 하나로 합쳐지고, 흩어져 있으면 32개가 된다.
최대 32배 차이다.

**지금 상태.** `instruction_cost(V_LD_GLOBAL_F32) = 100`, 끝이다. 주소를
보지 않는다.

```cpp
case Opcode::V_LD_GLOBAL_F32:
case Opcode::V_ST_GLOBAL_F32: return 100;
```

**따라오는 결과 세 가지:**

1. `V_LD_GLOBAL_VEC3_F32`가 명세만 있고 안 만들어진 이유가 이것이다.
   스칼라 로드 3개를 벡터 로드 1개로 바꿔도 **보여줄 것이 없다.**
   `DOC/09`의 "정정 — 지금 비용 모델로는 이 측정이 안 된다"가 이 이야기다.
2. 공유 메모리 스테이징이 95% 이긴 것이 "100 대 8"이라는 단순 산수로만
   설명된다. 진짜 이유(같은 캐시 라인을 32번 읽는 것 vs 한 번 읽어 재사용)는
   모델 밖이다.
3. AoS vs SoA, 정점 버퍼 레이아웃 같은 논의를 아예 못 한다.

**메우려면.** `instruction_cost(op)`가 opcode만 받는 구조를 깨야 한다. 워프
단위로 32레인의 주소를 모아 트랜잭션 수를 세는 함수가 필요하다:

```
warp_load_cost(addresses[32]) = 고유 캐시라인 수 × 라인당 비용
```

`execute()`가 스레드 하나를 받는 지금 구조와 맞지 않는다 — `step_warp`가
레인 루프를 돌기 **전에** 주소를 모아야 한다. `BARRIER`를 레인 루프 위에서
가로챈 것과 같은 모양의 변경이다.

**부작용.** `RESULTS.md`의 모든 수치가 다시 나온다. `render_bench`를 만들어둔
덕에 한 번 돌리면 갱신되지만, 문서의 서술(예: "-95.8%")도 같이 고쳐야 한다.

**규모.** 중간~큼. `gpurt/scheduler.cpp`, `gpurt/isa.cpp`, 테스트, 문서 전부.

### 2. Latency hiding / occupancy

**왜 중요한가.** 워프를 32개씩 묶고 SM에 여러 워프를 상주시키는 **이유 자체**다.
하나가 메모리를 기다리는 동안 다른 워프를 발행해서 ALU를 놀리지 않는다.

**지금 상태.** `myrt_launch`가 블록을 중첩 루프로 순차 실행한다.

```cpp
for (bz) for (by) for (bx) { ... scheduler_->run(program, tb, ...); }
```

명령어 비용이 고정이라 "기다린다"는 개념이 없다. 워프 스케줄러가 라운드로빈을
하지만, 그것이 **무엇을 숨기고 있는지**가 보이지 않는다.

**메우려면.** 명령어에 latency와 throughput을 따로 줘야 한다. 로드는 발행
비용 4, 결과 도착까지 400 같은 식으로. 그러면 스케줄러가 "결과를 기다리는
워프는 건너뛰고 다른 워프를 발행"할 수 있고, 상주 워프 수(occupancy)에 따라
총 시간이 달라지는 것이 나온다.

레지스터 사용량 ↔ occupancy 관계도 여기서 나온다. 스레드당 256 레지스터를
다 쓰면 상주 워프가 줄어든다는 실제 GPU의 트레이드오프.

**규모.** 큼. 스케줄러의 실행 모델을 바꾸는 일이다. 다만 **포트폴리오
관점에서 값이 제일 클 수도 있다** — divergence 다음으로 자주 묻는 개념이다.

### 3. Early-Z

**왜 중요한가.** depth complexity가 있는 씬에서 현대 래스터라이저를 빠르게
만드는 것의 큰 몫이다. 가려질 픽셀을 셰이딩 **전에** 버린다.

**지금 상태.** 픽셀이 자기를 덮는 모든 삼각형을 셰이딩하고 nearest만 남긴다.
지금 셰이딩이 무게중심 색이라 싸서 티가 안 나는데, `[7]` 조명이 들어간
레이 트레이서에서는 이미 삼각형마다 `normalize` 두 번을 낸다.

**메우려면.** depth 버퍼가 필요하고, 그러면 **두 픽셀이 같은 슬롯에 쓰는
경합**이 생긴다. 래스터라이저는 픽셀당 스레드 하나라 경합이 없지만,
early-Z를 하려면 삼각형을 앞에서 뒤로 정렬하거나 depth prepass를 따로 돌려야
한다. depth prepass 쪽이 우리 구조에 맞는다 — 런치 하나 더.

**규모.** 중간. 새 커널 하나 + 정렬. 원자적 연산은 필요 없다.

### 4. Input Assembler / 인덱스 버퍼 — `[9]`

**왜 중요한가.** 정점 하나를 보통 6개 삼각형이 공유한다. 하드웨어는 IA와
post-transform vertex cache가 자동으로 중복을 없앤다. 우리는 삼각형마다
정점을 펼쳐 넘기므로 **같은 정점을 6번 변환**한다. `V_MATVEC_MAT4_F32`가
비용 16이라 그대로 곱해진다.

**메우려면.** `Mesh { vertices, indices }`. 패스 1은 고유 정점만 변환한다.
호스트가 인덱스로 `ScreenTriangle`을 조립하는데, **이미 그러고 있다** —
`bin_triangles`가 화면 정점을 되읽어 타일마다 복사해 넣는다.

타일드/공유 경로는 비용이 0이고, 순회(walk) 경로만 종속 로드가 하나 는다.
ISA가 `global[reg[src0] + imm]`이라 인덱스를 로드해 주소로 쓰는 것은 가능하다.

**레이 트레이서는 여기서 빠진다.** 인덱싱이 사는 것은 패스 1 의 변환 중복인데
레이 트레이서에는 패스 1 이 없다. 삼각형당 로드가 9회에서 12회로 늘 뿐이다.

한 가지는 분명히 해둔다 — **실제 레이 트레이싱에도 정점 버퍼와 인덱스 버퍼는
있다.** DXR 의 `D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC` 와 Vulkan RT 가 둘 다
그 이름으로 받는다. 다른 것은 **언제 읽히는가**다.

```
래스터화   인덱스 버퍼를 draw 마다 IA 가 읽어 정점 셰이더를 먹인다
           매 프레임, 매 draw

RT         인덱스 버퍼를 BLAS 빌드 때 한 번 읽는다. 빌더가 삼각형을 자기
           형식으로 리프에 넣고, 광선은 그 리프를 읽는다 — 인덱스를 따라가지
           않는다
```

즉 RT 에서 인덱스 버퍼는 메모리와 빌드 입력으로 실재하고 유용하지만, **광선마다
간접 참조를 하지는 않는다.** post-transform vertex cache 에 해당하는 것이
없는 이유는 광선마다 도는 정점 변환이 없기 때문이다. (스키닝처럼 정점을 실제로
변환하는 패스가 따로 있으면 그쪽은 인덱스 이득을 본다 — BLAS 를 다시 짓기 전
별개 단계다.)

우리 레이 트레이서는 그보다 한 단계 아래다. **BVH 조차 없어서** 평탄한 삼각형
배열을 선형 순회하므로, 무차별 순회에 인덱스를 넣으면 종속 로드만 는다.

**다만 그것은 IA의 절반이다.** "고유 정점만 변환"은 캐시가 무한하다는 가정이고,
고정 기능 IA는 16~32엔트리 FIFO를 쓴다 — 밀려난 정점은 재변환되므로 **인덱스
순서가 성능을 바꾼다**(ACMR, Forsyth 최적화가 존재하는 이유). 우리처럼 화면
버퍼에 materialise 하면 그 문제가 사라지는 대신 버퍼와 패스가 든다. compute
기반 파이프라인과 mesh shader 가 실제로 그 선택을 한다.

유한 캐시였다면 몇 번 변환했을지는 **호스트에서 세기만 해도** 나온다. 실행
모델을 안 건드리고 "왜 materialise 인가"에 숫자를 붙일 수 있다.

**규모.** 작음. → `DOC/09` `[9]`, 초기 설계 기록 의 `[9a]`~`[9d]`

### 4b. BVH / 가속 구조 — 레이 트레이서의 타일 비닝

**왜 중요한가.** 래스터라이저에서 `[5b]` 타일 비닝이 한 일을 레이 트레이서에서
하는 것이 BVH다. 지금은 픽셀마다 **모든 삼각형**을 선형 순회한다.

```
우리        O(픽셀 × 삼각형)      정육면체 12개면 픽셀당 12번
실제 RT     BVH 순회             픽셀당 log(N) 노드
```

`render_bench` 에서 레이 트레이서가 `walk` 하고만 비교되고 `tiled` 와는
비교되지 않는 이유가 이것이다 — **비교할 대상이 없다.** 래스터라이저는 순회를
85% 줄인 변형이 둘 있는데 레이 트레이서는 순회밖에 없다.

**메우려면.** 호스트에서 BVH 를 짓고(중간 분할이면 충분하다), 커널이 스택으로
순회한다. 스택이 문제인데 — 레지스터 파일에 고정 깊이 배열을 두거나, 부모
포인터를 넣어 스택 없는 순회를 하거나 둘 중 하나다. 후자가 이 ISA 에 맞는다.
레지스터가 스레드당 256 개뿐이고, 순회 깊이가 씬에 따라 변하면 곤란하다.

**divergence 가 여기서 흥미로워진다.** 한 워프의 32 광선이 BVH 의 서로 다른
가지로 갈라진다 — 지금 우리가 재는 어떤 발산보다 심할 가능성이 높고, 실제
하드웨어가 광선 재정렬(ray reordering, Ada 의 SER)을 넣은 이유가 그것이다.

**규모.** 큼. 다만 `[e]` early-Z 보다 값이 클 수 있다. 이 프로젝트가 두
렌더러를 나란히 두는 구조라, 한쪽에만 가속이 있는 지금이 비대칭이다.

### 5. 클리핑

near plane 뒤의 삼각형을 자르지 않는다. 지금은 카메라가 항상 씬 밖에 있어서
드러나지 않지만, 카메라가 삼각형을 뚫고 지나가면 `w`가 0을 지나며 좌표가
발산한다.

**규모.** 작음. 다만 삼각형 하나가 둘이 될 수 있어서 정점 수가 런타임에
변한다 — 지금 구조가 그것을 상정하지 않는다.

### 5b. 비동기 — 실제 GPU 가 비동기인 지점

**지금 모든 런치가 동기다.** `myrt_launch` 가 블록을 끝까지 돌리고 반환하고,
`myrt_sync` 는 기다릴 것이 없어 출력만 한다.

실제 GPU 는 처음부터 비동기였다. 커널 런치가 즉시 반환하고, 스트림이 여러 개고,
복사가 계산과 겹친다. 그 위에 Ampere 의 `cp.async` 가 커널 **안**의 비동기를
더했다.

**쉬운 절반은 지금도 된다.** `kernel(args)` 가 런치 시점에 `Program` 을 만들고
상수를 전부 구우므로 그 프로그램은 자기완결적이다 — 큐에 넣어도 `args` 가 죽어도
된다.

**어려운 절반은 겹칠 것이 없다는 것이다.** 블록이 순차 실행되고 명령어 비용이
고정 숫자 하나라, "A 가 기다리는 동안 B 를 돌린다" 를 표현할 곳이 없다. 그래서
`[m4]`·`[m5]` 가 `[c]` 뒤에 온다.

**통계도 걸린다.** 지금 `stats_` 누적과 `elapsed_seconds_` 는 순차 실행을
전제한다. 스트림이 겹치면 divergence 가 섞이고 시간이 이중 계산된다.

**규모.** 큼. → 초기 설계 기록 `[m4]`, `[m5]`

### 6. 텍스처 / 샘플러

**안 한다.** 필터링과 밉맵은 그 자체로 큰 주제고, SIMT나 divergence에 대해
새로 말해주는 것이 적다. `DOC/09`의 "범위 밖"에 이미 들어 있다.

---

## 추가 자료

### 소프트웨어 래스터화

- Laine & Karras, **"High-Performance Software Rasterization on GPUs"**,
  *HPG 2011*.
  [research.nvidia.com/publication/high-performance-software-rasterization-gpus](https://research.nvidia.com/publication/2011-08_high-performance-software-rasterization-gpus)

  CUDA로 래스터라이저를 짜서 고정 기능과 비교한 논문. 우리가 서 있는 자리를
  정확히 다룬다 — 어디서 지고 어디서 이기는지.

- Fabian Giesen, **"A trip through the Graphics Pipeline 2011"**
  [fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/)

  고정 기능 블록이 실제로 무엇을 하는지 13편에 걸쳐 설명한다. 위 대조표의
  "하드웨어 블록" 열이 여기서 나왔다. IA와 post-transform vertex cache,
  Hi-Z, ROP 전부 다룬다.

- **Nanite** — Karis et al., *A Deep Dive into Nanite Virtualized Geometry*,
  SIGGRAPH 2021 Advances in Real-Time Rendering.
  [advances.realtimerendering.com/s2021/index.html](https://advances.realtimerendering.com/s2021/index.html)

  작은 삼각형에서 compute 래스터화가 고정 기능을 이기는 이유.

### 메모리

### Input Assembler / 정점 캐시

- Tom Forsyth, **"Linear-Speed Vertex Cache Optimisation"** (2006)
  [tomforsyth1000.github.io/papers/fast_vert_cache_opt.html](https://tomforsyth1000.github.io/papers/fast_vert_cache_opt.html)

  삼각형 순서만 바꿔 ACMR 을 내리는 알고리즘. 유한 캐시가 왜 문제인지가
  이 글의 전제에 다 들어 있다.

- Hoppe, **"Optimization of Mesh Locality for Transparent Vertex Caching"**,
  *SIGGRAPH 1999*.
  [dl.acm.org/doi/10.1145/311535.311565](https://dl.acm.org/doi/10.1145/311535.311565)

  ACMR 이라는 지표가 나온 논문.

### 레이 트레이싱

- **DirectX Raytracing (DXR) Functional Spec — Acceleration Structures**
  [microsoft.github.io/DirectX-Specs/d3d/Raytracing.html](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#acceleration-structure-properties)

  BLAS 가 정점 버퍼와 인덱스 버퍼를 받는다는 것, 그리고 빌드 결과가 불투명한
  구조라는 것이 여기 정의돼 있다. "RT 에 인덱스 버퍼가 없다"는 오해를 푸는
  1차 자료다.

- Meister et al., **"A Survey on Bounding Volume Hierarchies for Ray Tracing"**,
  *Eurographics 2021 STAR*.
  [diglib.eg.org/handle/10.1111/cgf142662](https://diglib.eg.org/handle/10.1111/cgf142662)

  BVH 빌드와 순회 전반. `[g]` 를 할 때 어떤 분할을 쓸지의 출발점.

- **NVIDIA Ada — Shader Execution Reordering**
  [developer.nvidia.com/blog/improve-shader-performance-and-in-game-frame-rates-with-shader-execution-reordering/](https://developer.nvidia.com/blog/improve-shader-performance-and-in-game-frame-rates-with-shader-execution-reordering/)

  광선이 BVH 의 서로 다른 가지로 갈라지며 생기는 divergence 를 하드웨어가
  어떻게 다루는지. `[g]` 가 이 프로젝트와 맞닿는 지점이다.

### 메모리

- **CUDA C++ Best Practices Guide — Coalesced Access to Global Memory**
  [docs.nvidia.com/cuda/cuda-c-best-practices-guide/](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

  `[d]`를 구현할 때 비용 모델이 따라야 할 규칙.

### Occupancy

- **CUDA C++ Programming Guide — Maximize Utilization / Occupancy**
  [docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-utilization)

  레지스터 사용량과 상주 워프 수의 관계. `[c]`의 목표.

---

## 현재 코드·검증 대조

- 구현: [docs/scope.md](../../docs/scope.md), [gpurt/gpu_spec.hpp](../../gpurt/gpu_spec.hpp)
- 검증: [test/unit/test_scheduler.cpp](../../test/unit/test_scheduler.cpp)
- 측정이 있는 단계는 해당 벤치마크의 소스와 `test/benchmark/RESULTS.md`를 함께
  기준으로 삼는다. 이 문서의 수치는 실행 당시의 결과이며, 현재 재현값은 실행 환경에
  따라 달라질 수 있다.

---

[← 이전 문서](./10_barrier_and_shared_memory.md) · [구현 노트 목차](./README.md) · [다음 문서 →](./12_warp_scheduling_guide.md)
