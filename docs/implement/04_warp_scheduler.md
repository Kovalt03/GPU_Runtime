# 4단계 — Warp Scheduler

## 목표

Warp를 순서대로 실행하고, 각 Instruction을 해석(decode)하여 Thread에 반영하는
실행 엔진을 구현한다. Divergence 통계(마스킹된 명령어 수)를 누적하여
포트폴리오 핵심 지표인 `divergence_rate()`를 계산할 수 있게 한다.

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `gpurt/scheduler.hpp` | WarpScheduler 클래스 선언 |
| `gpurt/scheduler.cpp` | Round-robin 스케줄링, ISA 실행 루프, 통계 구현 |
| `test/unit/test_scheduler.cpp` | GoogleTest — 명령어 실행·divergence·통계 검증 |

---

## 전체 실행 흐름

```
WarpScheduler::run(Program, Block&)
│
├─ readyQueue에 Block의 모든 Warp* 추가
│
└─ while (readyQueue not empty)
    │
    ├─ warp = readyQueue.front()  (Round-robin)
    ├─ readyQueue.pop()
    │
    ├─ instr = program[warp->pc]
    ├─ warp->pc++
    │
    ├─ for each lane 0..31:
    │   ├─ if !is_active(warp, lane)  → total_instructions++, masked_instructions++  (skip)
    │   └─ else                       → execute(instr, warp->threads[lane], block)
    │                                   total_instructions++
    │
    ├─ if instr.op == RET → 모든 lane이 inactive면 Warp 제거
    └─ else readyQueue.push(warp)     (다음 Warp에게 양보)
```

---

## 명령어별 실행 로직

```
VMUL_F32:   thread.regs[dst] = regs[src0] * regs[src1]
VADD_F32:   thread.regs[dst] = regs[src0] + regs[src1]
VDOT3:      thread.regs[dst] = regs[src0]*regs[src1]
                              + regs[src0+1]*regs[src1+1]
                              + regs[src0+2]*regs[src1+2]
VNORM3:     float len = sqrt(dot(src, src))
            regs[dst]   = regs[src0]   / len
            regs[dst+1] = regs[src0+1] / len
            regs[dst+2] = regs[src0+2] / len

LD_GLOBAL:  thread.regs[dst] = *reinterpret_cast<float*>(
                global_base + static_cast<int>(regs[src0]) + static_cast<int>(imm))
ST_GLOBAL:  *ptr = thread.regs[src1]  (동일 오프셋 계산)

LD_SHARED:  thread.regs[dst] = block.shared_mem[index]
ST_SHARED:  block.shared_mem[index] = thread.regs[src1]

BRA:        warp->pc += static_cast<int32_t>(imm) - 1  // pc는 이미 +1 됐으므로
BRA_DIV:    if (thread.regs[src0] != 0.0f) {
                thread.pc = warp->pc + static_cast<int32_t>(imm) - 1
                // 이 lane만 다른 PC → Warp diverge 시작
            } else {
                // 이 lane은 분기 안 함 → deactivate until reconverge
            }
RET:        thread.active = false
            deactivate(warp, lane)
```

---

## Divergence 통계

원시 카운터는 2개만 두고 나머지는 전부 파생시킨다. 두 수치가 서로 어긋날
여지를 없애기 위함.

```cpp
struct SchedulerStats {
    uint64_t warp_steps;       // 워프에 발행된 명령어 수
    uint64_t active_lane_ops;  // 실제로 일한 레인-명령어 수

    lane_slots()        = warp_steps * WARP_SIZE   // 구매한 용량
    masked_lane_slots() = lane_slots() - active_lane_ops
    divergence_rate()   = masked_lane_slots() / lane_slots()
};
```

**핵심: 스텝 하나는 참여 레인 수와 무관하게 항상 `WARP_SIZE`칸을 소모한다.**
레인 1개만 켜져 있어도 32칸을 산다 — 그것이 divergence가 비용인 이유다.

출력 형식 (Runtime::sync() 이후 자동 출력):
```
[STATS] divergence: 12.3%, throughput: 4.5 GIOPS
```

### 한계: 명령어 비용을 모두 1로 센다

`warp_steps += 1`은 어떤 명령어든 동일하다. 실제 하드웨어는 그렇지 않다
(NVIDIA Volta 이후 대략치):

| 명령어 | 지연 | 유닛 |
|--------|------|------|
| `V_ADD_F32`, `V_MUL_F32`, `V_FMA_F32` | ~4 사이클 | 일반 FP32 |
| `V_RCP_F32`, `V_SQRT_F32` | ~16–20 사이클 | SFU (개수도 적음) |
| `V_LD_SHARED_F32` | ~20–30 사이클 | 온칩 |
| `V_LD_GLOBAL_F32` | **~400–800 사이클** | DRAM |

우리 구현만 봐도 `V_NORM_VEC3_F32`(sqrt 1회 + 나눗셈 3회)와 `V_MOV_F32`가
같은 1로 집계된다.

**`divergence_rate`는 이 한계의 영향을 받지 않는다.** "발행된 용량 중 얼마가
놀았나"는 비율이라 명령어가 비싸든 싸든 동일하게 정의된다. 오히려 비싼
명령어에서 낭비의 절대량이 클 뿐이다.

**영향을 받는 것은 `throughput_giops()`다** (5단계). 명령어를 모두 똑같이
세면 sqrt만 도는 커널이 add만 도는 커널보다 빨라 보인다 — 실제로는 반대다.

→ **비용 모델은 5단계에서 도입한다.** `Opcode → uint32_t` 테이블과 카운터
하나면 되고, 그때가 이 값이 실제로 필요해지는 시점이다. 도입하면 7단계
벤치마크에서 "divergence 때문에 느림"과 "명령어가 비싸서 느림"을 분리해
보여줄 수 있고, 메모리 명령어에 큰 비용을 주면 coalescing 효과
(`V_LD_GLOBAL_VEC3_F32`, → `DOC/01` 확장 계획)가 수치로 드러난다.

> 참고로 **워프 스텝 *안*에서는** 이 문제가 없다. 한 스텝은 32레인이 모두
> 같은 명령어를 실행하므로 "제일 느린 레인을 기다리는" 상황이 생기지 않는다.
> 그리고 긴 지연은 실제 GPU에서도 **다른 워프로 전환해 숨긴다** — Round-robin이
> 워프당 한 스텝씩만 주고 큐 뒤로 보내는 것이 그 모델의 단순화 버전이다.

---

## Divergence 모델 — min-PC 재합류

초안은 `BRA_DIV`에서 **per-thread pc를 갱신하면서 동시에 마스크로 비활성화**
하는 방식이었다. 한 가지 일에 두 메커니즘이 겹쳐 "언제 다시 켜지는가"가
정의되지 않는다. 초안 자신도 "단순 구현을 위해 RET 기반 reconverge"라고만
적고 있어 실질적인 재합류가 없다.

**채택: `Thread::pc`를 유일한 진실로 두고, 마스크는 매 스텝 계산한다.**

```
워프 스텝 = 아직 살아있는 스레드들의 pc 중 최솟값을 고르고,
            그 pc에 있는 레인만 참여시킨다. 나머지는 이번 스텝에서 마스킹.
```

`Warp::pc`와 `Warp::active_mask`는 저장된 상태가 아니라 **매 스텝의 계산
결과**다 (`gpurt/thread.hpp` 주석에도 명시).

### 재합류가 저절로 된다

```
프로그램  0: BRA_DIV cond,+2   1: A   2: B   3: RET
          (짝수 레인만 조건 참)

스텝 1   pc 전부 0        → mask=0xFFFFFFFF (32)  BRA_DIV 실행
                            짝수는 pc=2, 홀수는 pc=1
스텝 2   pc {2, 1}        → 최솟값 1, mask=홀수만 (16)   ← 16레인 낭비
                            A 실행. 홀수도 pc=2
스텝 3   pc 전부 2        → mask=0xFFFFFFFF (32)   ← 저절로 다시 모임
```

reconvergence 스택이 필요 없고, **역방향 분기(루프)도 그냥 된다** — 되돌아간
스레드가 pc가 작으니 먼저 실행되고, 루프를 빠져나오면 나머지와 만난다.

---

## 데이터 구조

실제 인터페이스는 `gpurt/scheduler.hpp` 참조. 초안에서 바뀐 점:

| 초안 | 실제 | 이유 |
|------|------|------|
| `Block&` | `ThreadBlock&` | 3단계 개명 (`gpurt/memory.hpp`의 `Block`과 충돌) |
| `float* global_base` | `DeviceSpan{base, size}` | 크기를 모르면 범위 검사 불가 |
| `total/masked_instructions` | `warp_steps` / `active_lane_ops` | 위 통계 절 참조 |
| `execute_instruction(...)` | `step_warp()` + `execute()` | 워프 단위와 레인 단위를 분리 |

```cpp
struct DeviceSpan { uint8_t* base; size_t size; };

class WarpScheduler {
public:
    void run(const Program& program, ThreadBlock& block, DeviceSpan global);

    const SchedulerStats& stats() const;
    double divergence_rate() const;
    void   reset_stats();

private:
    std::queue<Warp*> ready_queue_;   // 포인터: Warp 하나가 ~32KB
    SchedulerStats    stats_;

    bool step_warp(const Program&, Warp&, ThreadBlock&, DeviceSpan);
    void execute(const Instruction&, Thread&, ThreadBlock&, DeviceSpan);
};

// 테스트에서 직접 호출하는 검증 헬퍼
size_t decode_address(float value, const char* what);
void   require_register_range(uint32_t reg, uint32_t count, const char* what);
```

### 주소 규약

레지스터 값은 **바이트 오프셋**이다 (`device_alloc`이 주는 값과 같은 단위).
`shared_mem`은 `float` 배열이지만 바이트로 재해석해 같은 규칙을 쓴다.

```
decode_address()      float → size_t. 음수·NaN·무한대·비정수 거부
require_f32_access()  4바이트 정렬 + 범위 (load_f32/store_f32 내부)
```

정렬 검사가 `decode_address`가 아니라 `require_f32_access`에 있는 이유:
분기 오프셋도 `decode_address`를 지나가는데 명령어 인덱스는 4의 배수일 이유가
없다.

---

## 구현 흐름

```
[1] gpurt/scheduler.hpp 작성
    └─ WarpScheduler 클래스 선언
    └─ Stats 구조체 (혹은 멤버로 통합)

[2] gpurt/scheduler.cpp 작성
    └─ run(): readyQueue 초기화 → 메인 실행 루프
    └─ execute_instruction(): opcode switch-case
       └─ 산술: VMUL, VADD, VDOT3, VNORM3
       └─ 메모리: LD/ST GLOBAL/SHARED (reinterpret_cast + 범위 검사)
       └─ 제어: BRA (warp pc 수정), BRA_DIV (per-thread pc), RET
    └─ divergence_rate(): masked / total 계산

[3] test/unit/test_scheduler.cpp 작성 (TDD)
    └─ TEST(Scheduler, VmulF32)           — 단일 Warp, VMUL 실행 결과 검증
    └─ TEST(Scheduler, VaddF32)
    └─ TEST(Scheduler, Vdot3Result)       — 내적 결과 정확도 확인 (±1e-5)
    └─ TEST(Scheduler, Vnorm3Unit)        — 정규화 후 길이 == 1.0
    └─ TEST(Scheduler, LdStGlobal)        — 전역 메모리 읽기/쓰기 후 값 확인
    └─ TEST(Scheduler, LdStShared)        — 공유 메모리 읽기/쓰기 후 값 확인
    └─ TEST(Scheduler, BranchTaken)       — BRA로 PC 이동 확인
    └─ TEST(Scheduler, DivergenceStats)   — BRA_DIV 후 divergence_rate() > 0
    └─ TEST(Scheduler, AllRetInactive)    — 모든 스레드 RET 후 Warp 제거 확인
```

---

## 설계 결정 및 이유

**Round-robin 스케줄링**
: 구현 단순성 우선. 실제 GPU는 레이턴시 히딩을 위해 ready warp를 우선하는 greedy 정책을 사용하지만, 시뮬레이터에서는 divergence 통계 측정이 목적이므로 Round-robin으로 충분.

**`std::queue<Warp*>` (포인터 저장)**
: Warp 객체를 복사하지 않고 Block 내 Warp를 직접 수정. Block의 수명이 run() 호출 동안 유지됨을 전제.

**BRA_DIV의 per-thread pc 갱신**
: Warp.pc는 "다음 실행할 공통 PC", Thread.pc는 diverge 후 개별 경로. Reconverge 시점(모든 활성 스레드가 같은 PC에 도달)에 Warp.pc를 갱신하고 full mask 복원 — 단순 구현을 위해 RET 기반 reconverge 사용.

**전역 메모리 접근 = raw pointer 산술**
: `float* global_base`를 Scheduler에 전달하고 레지스터 값 + imm으로 오프셋 계산. 범위 초과 시 `std::runtime_error`.

---

## 다음 단계

→ [05_runtime_api.md](05_runtime_api.md) — Public Runtime API 및 통계 집계

---

## 현재 코드·검증 대조

- 구현: [gpurt/scheduler.hpp](../../gpurt/scheduler.hpp), [gpurt/scheduler.cpp](../../gpurt/scheduler.cpp)
- 검증: [test/unit/test_scheduler.cpp](../../test/unit/test_scheduler.cpp)
- 측정이 있는 단계는 해당 벤치마크의 소스와 `test/benchmark/RESULTS.md`를 함께
  기준으로 삼는다. 이 문서의 수치는 실행 당시의 결과이며, 현재 재현값은 실행 환경에
  따라 달라질 수 있다.

---

[← 이전 문서](./03_thread_warp.md) · [구현 노트 목차](./README.md) · [다음 문서 →](./05_runtime_api.md)
