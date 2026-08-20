# 3단계 — Thread / Warp / ThreadBlock 구조체

## 목표

SIMT(Single Instruction Multiple Threads) 실행 모델의 핵심 자료구조를 정의한다.
Warp 내 32개 스레드가 동일 명령어를 실행하되, `activeMask`로 개별 스레드의
활성 여부를 제어하여 Warp divergence를 시뮬레이션한다.

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `gpurt/thread.hpp` | Thread, Warp, Block 구조체 선언 |
| `gpurt/thread.cpp` | 초기화 헬퍼, activeMask 조작 유틸리티 구현 |

> Thread/Warp/Block은 순수 데이터 구조체. 실행 로직은 Scheduler(4단계)에 위임.

---

## SIMT 실행 모델 개요

```
ThreadBlock (CTA)
├── Warp 0  [Thread 0..31]  ← activeMask = 0xFFFFFFFF (전체 활성)
├── Warp 1  [Thread 32..63]
│   ├── Thread 32  active=true   regs[0..255]  pc=12
│   ├── Thread 33  active=false  ← BRA_DIV로 분기, 이 Warp에서 비활성화
│   └── ...
└── sharedMem[4096 bytes]   ← Warp들이 공유
```

**Warp divergence 발생 시나리오**:
```
// BRA_DIV src0, offset
// Warp 내 스레드마다 reg[src0] 값이 다를 수 있음

Thread 0: reg[src0] = 1.0f → 분기 실행
Thread 1: reg[src0] = 0.0f → 분기 미실행 (비활성화)
Thread 2: reg[src0] = 1.0f → 분기 실행
...

→ 같은 Warp 내 스레드가 서로 다른 경로를 탐 → divergence
→ 비활성 스레드는 activeMask로 마스킹, 결과를 레지스터에 반영하지 않음
```

---

## 데이터 구조

```cpp
// gpurt/thread.hpp

struct Thread {
    std::array<float, 256> regs;  // 범용 레지스터 파일 (float 전용)
    uint32_t pc;                  // Program Counter (Program 내 인덱스)
    bool     active;              // 현재 실행 중인지 여부
};

struct Warp {
    std::array<Thread, 32> threads;
    uint32_t pc;           // Warp 공통 PC (diverge 전까지 동일)
    uint32_t active_mask;  // 비트 i = 스레드 i의 활성 여부 (1=active)
};

// Block 이 아니라 ThreadBlock 이다 — memory.hpp 가 이미 전역 `struct Block`
// (할당기의 블록)을 정의하고 있고, 4단계 Scheduler 는 두 헤더를 모두 include
// 한다. CUDA 자체 용어도 "thread block" 이라 이쪽이 더 정확하다.
struct ThreadBlock {
    std::vector<Warp>        warps;
    std::array<float, 4096>  shared_mem;  // 4096 floats = 16KB (shared memory)
    uint32_t                 block_id;
};
```

### 레지스터 파일 선택 이유

| 항목 | 결정값 | 이유 |
|------|--------|------|
| 레지스터 수 | 256개 | PTX 최대 255개 레지스터 + 여유 1개 |
| 타입 | `float` 전용 | 시뮬레이터 단순화. int 연산은 float 비트 재해석으로 표현 |
| Warp 크기 | 32 threads | 실제 NVIDIA GPU와 동일 |
| Shared memory | 4096 floats | 16KB — SM당 shared mem 최솟값 기준 |

---

## activeMask 동작

```
active_mask = 0b00000000_00000000_11111111_11111111
                                  ^^^^^^^^^^^^^^^^
                                  하위 16 스레드만 활성

// 비트 조작 예시 (gpurt/thread.cpp 헬퍼)
bool is_active(const Warp& w, uint32_t lane) {
    return (w.active_mask >> lane) & 1u;
}

void deactivate(Warp& w, uint32_t lane) {
    w.active_mask &= ~(1u << lane);
}

void activate(Warp& w, uint32_t lane) {
    w.active_mask |= (1u << lane);
}
```

---

## 구현 흐름

```
[1] gpurt/thread.hpp 작성
    └─ struct Thread
    └─ struct Warp
    └─ struct Block
    └─ activeMask 조작 인라인 헬퍼 (is_active, deactivate, activate)

[2] gpurt/thread.cpp 작성
    └─ lane_mask(uint32_t lane_count) → 하위 lane_count 비트 마스크
       └─ ⚠ (1u << lane_count) - 1 은 lane_count == 32 에서 UB (아래 참조)
    └─ active_lane_count(const Warp&) → popcount(active_mask)
    └─ make_warp(uint32_t lane_count) → Warp 초기화
       └─ threads 는 기본 멤버 초기화로 이미 0 / pc=0 / active=true
       └─ active_mask = lane_mask(lane_count)
    └─ make_block(uint32_t warp_count, uint32_t block_id) → ThreadBlock 초기화
       └─ warps.resize(warp_count) 후 각 Warp 의 active_mask 설정
       └─ shared_mem 은 기본 멤버 초기화로 이미 0
```

### ⚠ `(1u << lane_count) - 1` 은 쓰면 안 된다

32비트 값을 32칸 이상 시프트하는 것은 **정의되지 않은 동작**이다. 그리고
`lane_count == 32` 는 예외 상황이 아니라 **모든 워프의 기본값**이다.

```
lane_count= 1  (1u<<n)-1 = 0x00000001   올바른 값 = 0x00000001
lane_count=16  (1u<<n)-1 = 0x0000FFFF   올바른 값 = 0x0000FFFF
lane_count=31  (1u<<n)-1 = 0x7FFFFFFF   올바른 값 = 0x7FFFFFFF
lane_count=32  (1u<<n)-1 = 0x7FFFFFFF   올바른 값 = 0xFFFFFFFF   ← 불일치
```

(AppleClang 16, `-O2` 실측. UB이므로 컴파일러·최적화 수준에 따라 달라진다.)

31번 레인이 영원히 비활성이 되고, 그 워프는 32개가 아니라 31개로만 돈다.
divergence 통계도 전부 어긋난다. **전폭 케이스를 분리해서 처리해야 한다.**

```cpp
uint32_t lane_mask(uint32_t lane_count) {
    if (lane_count >= WARP_SIZE) return 0xFFFFFFFFu;
    return (1u << lane_count) - 1u;
}
```

> 초안에는 Thread/Warp 전용 테스트를 두지 않고 Scheduler 테스트(4단계)에서
> 통합 검증한다고 되어 있었으나, `test/unit/test_thread.cpp` 를 따로 두었다.
> 비트 마스크는 **31번 레인 하나에서만 조용히 틀리는** 종류의 코드라,
> 스케줄러까지 가서 발견하면 원인이 마스크인지 실행 로직인지 구분되지 않는다.

---

## 설계 결정 및 이유

**`active_mask` uint32_t 비트필드**
: Warp 크기가 32로 고정이므로 uint32_t 1개로 표현 가능. 마스킹 연산이 단일 비트 AND/OR로 처리되어 빠르다.

**Thread.pc vs Warp.pc**
: 정상 실행 시 모든 스레드가 동일 PC를 공유(Warp.pc). Diverge 시 각 Thread.pc가 독립적으로 진행. Scheduler가 reconverge(재합류) 시점에 Warp.pc를 갱신한다.

**`shared_mem`을 ThreadBlock에 배치**
: Warp들이 같은 Block 안에서 shared memory를 공유하는 SIMT 모델의 직접 반영.

---

## 다음 단계

→ [04_warp_scheduler.md](04_warp_scheduler.md) — Warp 스케줄러 및 실행 엔진

---

## 현재 코드·검증 대조

- 구현: [gpurt/thread.hpp](../../gpurt/thread.hpp), [gpurt/thread.cpp](../../gpurt/thread.cpp)
- 검증: [test/unit/test_thread.cpp](../../test/unit/test_thread.cpp)
- 측정이 있는 단계는 해당 벤치마크의 소스와 `test/benchmark/RESULTS.md`를 함께
  기준으로 삼는다. 이 문서의 수치는 실행 당시의 결과이며, 현재 재현값은 실행 환경에
  따라 달라질 수 있다.

---

[← 이전 문서](./02_memory_model.md) · [구현 노트 목차](./README.md) · [다음 문서 →](./04_warp_scheduler.md)
