# 16 — `[m4]` 다중 SM · 스트림 · indirect launch
> `DOC/13` 가 `[m3]`·`[d]` 에 대해 그랬듯 "어떻게" 를 적는다. 지금까지 중 가장 큰
> 항목이고, **런타임 API 를 바꾸는 첫 항목**이다.
>
> `[m4a]`(SM 다중 + 점유율)는 **구현 완료**, `[m4b]`(스트림 + indirect)는 남았다.

---

## 0. 용어 — 이 문서가 쓰는 다섯 단어

혼동하기 쉬워서 먼저 못 박는다. 왼쪽이 CUDA 의 말이고 가운데가 우리 코드다.

| 말 | 우리 타입 | 무엇인가 | 누가 만드나 |
|---|---|---|---|
| **lane** | `Thread` | 레지스터 256개와 자기 `pc` 를 가진 실행 하나 | `make_block` |
| **warp** | `Warp` | lane 32개. **명령을 함께 발행하는 단위** | `make_block` |
| **block** | `ThreadBlock` | warp 여럿 + 공유 메모리 4096 float + `block_id` | 런타임이 런치마다 |
| **grid** | `dim3 grid` | 이번 런치가 돌릴 **블록의 집합**. 3차원 좌표 | 호출자가 런치에 명시 |
| **SM** | `Unit` (run_grid 안) | 블록을 **얹어놓고 돌리는 자리**. L1 과 발행 슬롯을 가짐 | 스케줄러가 `SMConfig` 대로 |

```
grid  (2 x 32 블록)                       SM 0            SM 1
┌──────┬──────┐                      ┌──────────┐   ┌──────────┐
│ blk0 │ blk1 │                      │ slot: b0 │   │ slot: b1 │
│ blk2 │ blk3 │   ──── 배분 ────►    │ slot: b2 │   │ slot: b3 │
│ ...  │ ...  │                      │  L1 (자기 것) │  L1 (자기 것) │
└──────┴──────┘                      └──────────┘   └──────────┘
                                          └──── 공유: L2, 장치 메모리 ────┘
```

**grid 는 "무엇을 할 일인가", SM 은 "어디서 할 것인가"** 다. 둘은 독립이다 — 블록
64개짜리 grid 를 SM 1개로 돌리면 64번 줄서고, 4개로 돌리면 16번 줄선다. **일의 총량은
같고 걸린 시간만 다르다.** 이것이 `[m4a]` 의 전부이고, 테스트
`MoreSMsFinishSoonerWithoutDoingLessWork` 가 그것을 못 박는다.

### 왜 "SM" 이라는 층이 필요했나

`[m3]` 에서 지연 모델이 들어왔을 때, 워프가 서로의 대기를 가려주는 것은
**블록 안에서만** 됐다. `myrt_launch` 가 블록을 하나씩 끝까지 돌렸기 때문이다.

```
지금까지    for (block : grid) { L1 비우고, 이 블록이 끝날 때까지 돌린다 }
```

실제 하드웨어에서 점유율(occupancy)이란 **한 SM 에 블록이 몇 개나 동시에 얹히는가**
이고, 그것이 곧 "가릴 수 있는 워프가 몇 개인가" 다. 블록을 한 번에 하나만 돌리는
기계에는 점유율이라는 개념이 아예 없다. `README` 가 "Multiple SMs, occupancy —
**absent**" 라고 적어둔 것이 이 뜻이었다.

---

## 1. 상태의 소유자가 바뀐다

| | 전 | 후 | 왜 |
|---|---|---|---|
| **L1** | 스케줄러에 하나, 블록 시작마다 `clear()` | **SM 마다 하나**, 블록이 오가도 유지 | 하드웨어에서 L1 은 SM 의 것이다 |
| **ready 큐** | 블록 하나의 워프들 | 없앰 — 슬롯·워프를 커서로 순회 | 상주 블록이 여럿이면 큐 하나로는 누구 것인지 모른다 |
| **`cycles`** | 블록마다 재서 **합산** | 런치 전체의 **벽시계** | 겹쳐 도는데 합하면 이중 계산이다 |
| **`stall_steps`** | 아무도 발행 못 한 사이클 | **일을 들고 있는데** 못 쏜 SM 마다 합산 | 블록을 못 받은 SM 은 대기가 아니라 미고용 |
| **shared_mem** | 블록 소유 | 그대로 | |
| **L2** | 장치 소유 | 그대로 | SM 들이 공유하는 유일한 것 |

**L1 이 SM 의 것이 되면서 질문이 하나 생긴다.** 같은 SM 에 얹힌 블록 둘이 같은
데이터를 읽으면 두 번째는 첫 번째가 끌어온 라인을 찾는다(이득). 다른 데이터를 읽으면
서로 밀어낸다(스래싱). 둘 다 잴 수 있다 — 순차 walk 은 블록마다 **같은** 삼각형을
읽고, 타일드는 블록마다 **다른** 타일을 읽는다. 정확히 두 경우다.

---

## 2. 점유율이 처음으로 계산된다

```
residency = min(blocks_per_sm,
                warp_slots_per_sm   / warps_per_block,
                shared_bytes_per_sm / shared_bytes_per_block)
```

`GPUSpec::residency()` 하나에 들어 있고, 셋 중 무엇이 걸리는지가 자명하지 않다는
것이 실제로 occupancy calculator 라는 물건이 존재하는 이유다. 테스트
`SharedMemoryAndWarpCountAreWhatLimitResidency` 가 셋을 각각 걸리게 해서 확인한다.

**세 번째 축이 핵심이다.** `ThreadBlock` 은 4096 float 을 통째로 들고 있지만 그건
"쓸 수 있는 최대"지 "쓰겠다고 선언한 양"이 아니다. 점유율을 정하는 것은 후자이고,
그래서 **런치가 선언한다**:

```cpp
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_bytes = 0;   // 이 커널이 블록당 쓰겠다는 양
};
```

이것이 있어야 **공유 메모리 스테이징의 진짜 대가**가 보인다. `[5c]` 는 글로벌 로드를
공유 로드로 바꿔 이겼고 `[d]` 에서 그 이득이 사라졌는데, 하드웨어에서 그 대가는
**점유율 하락**이다. 지금까지 모델에 그 대가를 표현할 곳이 없었다.

---

## 3. 구현 — `run_grid` 해부

전부 `WarpScheduler::run_grid(program, global, shared_bytes, next_block)` 안에 있다.

### 3.1 자료 구조

```cpp
struct Slot {                                  // SM 위의 블록 한 자리
    std::unique_ptr<ThreadBlock> block;        // 비었으면 nullptr
    std::vector<bool> live;                    // 워프별 생존
    size_t remaining = 0;                      // 살아있는 워프 수
    size_t cursor = 0;                         // 다음에 볼 워프
};

struct Unit {                                  // SM 하나
    std::vector<Slot> slots;                   // 최대 residency 개
    size_t cursor = 0;                         // 다음에 볼 슬롯
};
```

`unique_ptr` 인 이유: 슬롯 벡터가 재할당돼도 **블록의 주소가 안 움직여야** 한다.
`step_warp` 이 `ThreadBlock&` 를 받고, 워프 참조도 그 안을 가리킨다.

커서가 슬롯과 워프 양쪽에 있는 이유: SM 은 **블록들 사이에서도** 라운드로빈해야
한다. 한 블록의 워프만 계속 돌리면 co-residency 가 아무 의미도 없다.

### 3.2 residency 는 첫 블록이 정한다

```cpp
auto first = std::make_unique<ThreadBlock>();
if (!next_block(*first)) return;               // 빈 grid 는 0 사이클
per_sm = spec_.residency(first->warps.size(), shared_bytes);
```

워프 수를 알아야 계산이 되고, 워프 수는 블록을 만들어봐야 안다. 격자 안의 블록은
전부 같은 모양이므로 첫 번째로 정해도 된다 — 이 가정이 깨지는 날은 커널마다 블록
모양이 다른 런치가 생기는 날이고, 그때는 여기가 먼저 깨진다.

### 3.3 블록 공급 — 미리 다 만들지 않고, 넓게 뿌린다

```cpp
using NextBlock = std::function<bool(ThreadBlock&)>;
```

런타임이 "블록 i 를 지어서 좌표를 심는" 법을 알고, 스케줄러는 "슬롯이 비었다"는
것을 안다. 그래서 **콜백**이다. 격자를 통째로 만들지 않는 이유는 크기다 — 워프
하나가 32 KB 이고 256×256 격자면 블록 2,048 개다.

`fill_all()` 이 **너비 우선**으로 뿌린다 — 어느 SM 도 두 번째 블록을 받기 전에 모든
SM 이 첫 블록을 받는다. SM 하나를 꽉 채우고 넘어가는 것과 같아 보이지만 아니다:
`sphere.obj` 를 128×64 로 그리면 블록 256개인데, SM 108개가 32개씩 들 수 있는 기계에
깊이 우선으로 주면 **앞의 8개 SM 이 전부 가져가고 100개가 논다.** 실측으로 80배 대
250배였다. 하드웨어의 work distributor 가 넓게 뿌리는 이유도 같다.

격자가 마르면 `grid_exhausted` 가 서고 다시는 안 부른다.

### 3.4 메인 루프 — 한 사이클에 일어나는 일

```
1. 예산 확인 (now > cycle_budget_ → throw)
2. SM 마다:
     a. active_l1_ = 이 SM 의 L1        ← cache_lookup 이 이걸 쓴다
     b. 슬롯을 커서 순으로 돌며, 그 안의 워프를 커서 순으로 본다
          죽었거나 배리어에 있으면 건너뛴다
          ready_at > now 면 soonest 후보로만 두고 건너뛴다
          그 외 → step_warp 한 번. 발행됐으면 이 SM 의 차례 끝
     c. 아무것도 못 쐈으면: 이 SM 의 슬롯 중 "전원 배리어, 대기자 없음" 인
        블록의 배리어를 연다. 그리고 idle++
     d. 다 끝난 슬롯을 비우고 fill 로 다음 블록을 당긴다
3. 하나라도 발행됐으면 → stall_steps += idle; ++now; 다음 사이클
4. 아무것도 상주하지 않으면 → 종료
5. 기다리는 워프가 있으면 → now 를 그 시각으로 점프, 건너뛴 사이클 × SM 수를 stall 로
6. 배리어를 열었거나 블록을 새로 실었으면 → 시계는 안 돌리고 다음 사이클
7. 그 외 → 종료 (더 이상 달라질 것이 없다)
```

**한 사이클 = SM 하나당 명령 하나.** SM 넷이면 한 사이클에 넷이 나간다. 그래서
같은 일이 더 적은 사이클에 끝나고, `warp_steps` 는 한 자리도 안 변한다.

### 3.5 시계가 언제 도는가

이 표가 이 구현에서 가장 미묘한 부분이다.

| 사건 | `now` | 왜 |
|---|---|---|
| 명령 발행 | +1 | 사이클의 정의 |
| 은퇴한 워프 발견 | 0 | 발행이 아니다. 이전 큐 방식도 시간을 안 썼다 |
| 배리어 열림 | 0 | 배리어 뒤 워프는 **다음** 사이클에 쏜다. 여기서 시간을 매기면 Ignored 의 정의(`cycles == warp_steps`)가 깨진다 |
| 슬롯에 새 블록 적재 | 0 | 배정은 일이 아니다 |
| 전원 대기 | 가장 이른 `ready_at` 로 점프 | 그 사이 아무도 못 쏜다 |

이 넷을 각각 틀렸을 때 무엇이 깨지는지 실제로 겪었다 — 은퇴 발견에서 `break` 했더니
런치가 조기 종료했고, 배리어 열림에 `++now` 했더니
`ModelledSchedulingIssuesTheSameWorkAsIgnored` 가 1 사이클 어긋났으며, 블록 적재를
"변화 없음"으로 봤더니 블록 하나만 돌고 끝났다.

### 3.6 배리어는 블록 단위다

```cpp
if (at_barrier && !waiting) { release_barrier(*slot.block); }
```

한 SM 에 블록 둘이 얹혀 있어도 **배리어는 각자의 것**이다. 블록 A 가 배리어에서
기다리는 것과 블록 B 가 도는 것은 무관하다. 조건이 "이 블록의 살아있는 워프 중
배리어에 있는 것이 있고, 결과를 기다리는 것은 없다" 인 이유는 후자가 아직 **도착할 수
있기** 때문이다 — 늦는 워프를 두고 열면 배리어가 배리어가 아니게 된다.

### 3.7 예외가 나가도 블록은 돌려준다

```cpp
struct Handback { ... ~Handback() { 첫 상주 블록을 last_block_ 으로 } } handback{...};
```

예산을 만난 커널은 **던진다**. 그런데 `run()` 의 호출자는 그 블록의 주인이고,
교착 테스트는 **실패한 블록의 pc 를 읽어서** 어떤 실패였는지 말한다. 그래서 소멸자로
돌려준다. 이걸 빼먹었을 때 그 테스트가 세그폴트로 잡혔다.

### 3.8 `run()` 은 이제 특수 케이스다

```cpp
void WarpScheduler::run(program, block, global)   // 블록 하나짜리 grid
```

블록을 한 번만 넘기는 `NextBlock` 을 만들어 `run_grid` 를 부른다. 소유권을 넘겼다가
`GiveBack` 소멸자로 되돌려받는다 — 복사하지 않는 이유는 32 KB 이고, 소멸자인 이유는
던지는 경로에서도 돌려줘야 하기 때문이다.

**루프가 하나로 합쳐졌다.** `run_modelled` 은 사라졌다. Ignored 는 "모든 지연이 0인
Modelled" 이므로 같은 루프가 둘 다 서고, 그 등가성은 원래부터 테스트가 주장하던 것이다
(`ModelledSchedulingMatchesIgnoredWhenNothingWaits`). 대신 `step_warp` 끝에서
Ignored 면 `issued_latency_ = 0` 으로 눌러야 한다 — 예전 큐 경로는 `ready_at` 을
아예 안 읽어서 이 구분이 필요 없었다.

---

## 4. 런타임 쪽 변경

```cpp
void myrt_launch(KernelFunc, dim3 grid, dim3 block, void** args);        // 그대로
void myrt_launch(KernelFunc, const LaunchConfig&, void** args);          // 새로
void myrt_set_sm_config(const SMConfig&);
void myrt_set_spec(const GPUSpec&);
const GPUSpec& myrt_spec() const;
```

기존 4인자 형태를 남긴 이유는 커널·테스트·벤치가 전부 그걸 부르기 때문이고,
`shared_bytes` 를 기본 인자로 끼워넣지 않은 이유는 **잊기 쉬운 것보다 보이는 것**이
낫기 때문이다.

**블록 짓기가 콜백 안으로 들어갔다.** 예전에는 3중 루프가 블록을 만들고 좌표를 심고
바로 돌렸다. 지금은 같은 코드가 `build` 람다 안에 있고, 스케줄러가 필요할 때 부른다.

**통계 회계.** `reset_stats()` 가 블록마다 → **런치마다** 로 옮겼다. 총량 카운터는
어차피 합이라 결과가 같지만, `cycles` 는 겹쳐 도는 순간 합이 뜻을 잃는다.

---

## 5. 이음매와 검증

**기본값이 오늘을 재현한다.** `sm_count = 1, blocks_per_sm = 1` 이면 블록이 하나씩
끝까지 돌고 벽시계가 곧 합이다. `render_bench` 가 **바이트 단위로 동일**한 것으로
확인했다.

새 테스트 넷:

| 테스트 | 무엇을 못 박나 |
|---|---|
| `MoreSMsFinishSoonerWithoutDoingLessWork` | 사이클은 줄고 `warp_steps`·`weighted_lane_ops` 는 한 자리도 안 변한다 |
| `OneBlockCannotUseASecondSM` | 블록이 하나면 기계를 넓혀도 아무것도 안 변한다 |
| `SharedMemoryAndWarpCountAreWhatLimitResidency` | 세 한계가 각각 걸린다. 아무것도 못 맞추는 블록도 혼자서는 돈다 |
| `BlocksSharingAnSMShareItsL1` | L1 이 블록마다 지워지지 않는다 |

기존 테스트 둘을 고쳤다. 스핀 교착 테스트가 pc 를 **4** 로 못 박고 있었는데 4~6
범위로 바꿨다 — 예산이 루프의 어느 명령에서 터지는가는 스케줄러 부기이지 주장이
아니다. 그리고 유휴 SM 을 `stall_steps` 에 안 세기로 했다.

---

## 7. 구현에서 드러난 회귀

`render_bench` 를 돌려보니 walk 이 **+30,720 lane-ops** 늘어 있었다. 추적해보니
`[m4a]` 가 아니라 **early-Z·조명 커밋에서 들어온 것**이었고, 그때 벤치를 다시 안
돌려서 아무도 몰랐다.

| 원인 | 비용 |
|---|---|
| 깊이 주소 계산을 `DepthUse::None` 에서도 방출 | 10,240 |
| 조명 상수 10개를 `Barycentric` 에서도 방출 | 20,480 |

둘 다 `gpurt/pipeline/raster.hpp` 가 스스로 적어둔 원칙 — "선택된 형태만 명령 스트림에 들어간다" —
위반이다. 방출을 가드로 막아 **10,885,760 으로 정확히 복귀**했다.

**교훈은 절차 쪽이다.** 커널을 건드린 커밋은 `render_bench` 를 다시 돌려야 한다.
`ctest` 는 프레임이 같은지만 보고 **얼마나 드는지는 안 본다.**

---

## 현재 코드·검증 대조

- 구현: [gpurt/scheduler.cpp](../../gpurt/scheduler.cpp), [gpurt/gpu_spec.hpp](../../gpurt/gpu_spec.hpp)
- 검증: [test/unit/test_scheduler.cpp](../../test/unit/test_scheduler.cpp), [test/benchmark/src/occupancy_bench.cpp](../../test/benchmark/src/occupancy_bench.cpp)
- 측정이 있는 단계는 해당 벤치마크의 소스와 `test/benchmark/RESULTS.md`를 함께
  기준으로 삼는다. 이 문서의 수치는 실행 당시의 결과이며, 현재 재현값은 실행 환경에
  따라 달라질 수 있다.

---

[← 이전 문서](./15_early_z.md) · [구현 노트 목차](./README.md) · [다음 문서 →](./17_streams.md)
