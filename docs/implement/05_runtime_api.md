# 5단계 — Runtime API

## 목표

사용자(커널 개발자)가 직접 호출하는 퍼블릭 인터페이스를 제공한다.
CUDA Runtime API(`cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel`)와 유사한
구조로 설계하여 포트폴리오에서 설계 의도를 명확히 전달한다.

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `gpurt/runtime.hpp` | MyGPURuntime 클래스, dim3, KernelFunc 선언 |
| `gpurt/runtime.cpp` | MemoryManager · WarpScheduler 조율, 통계 집계 구현 |
| `test/unit/test_runtime.cpp` | GoogleTest — 전체 레이어 통합 테스트 |

---

## API 설계

```cpp
// gpurt/runtime.hpp

using KernelFunc = std::function<Program(void**)>;

struct dim3 {
    uint32_t x = 1, y = 1, z = 1;
};

class MyGPURuntime {
public:
    explicit MyGPURuntime(size_t device_mem_size = 64 * 1024 * 1024);  // 기본 64MB

    // 메모리 관리
    void*  myrt_malloc(size_t size);
    void   myrt_free(void* ptr);
    void   myrt_memcpy(void* dst, const void* src, size_t size, Direction dir);

    // 커널 실행
    void   myrt_launch(KernelFunc kernel, dim3 grid, dim3 block, void** args);

    // 동기화 (시뮬레이터는 동기 실행이므로 통계 출력 트리거)
    void   myrt_sync();

    // 통계 조회
    float  divergence_rate()    const;
    double throughput_giops()   const;

private:
    std::unique_ptr<MemoryManager>  mem_;
    std::unique_ptr<WarpScheduler>  scheduler_;

    // 통계 누적
    uint64_t   total_instructions_  = 0;
    uint64_t   masked_instructions_ = 0;
    double     elapsed_ns_          = 0.0;

    void print_stats() const;  // [STATS] 출력
};
```

---

## 레이어 연결 구조

```
myrt_launch(kernel, grid={1,1,1}, block={1,1,1}, args)
│
├─ kernel(args) → Program  (커널 함수가 ISA 명령어 시퀀스 반환)
│
├─ for each block in grid × block:
│   ├─ Block blk = make_block(block.x * block.y * block.z / 32)
│   └─ scheduler_->run(program, blk, global_mem_base)
│
└─ 통계 누적: total += scheduler->total_instructions()
              masked += scheduler->masked_instructions()
              elapsed += measure(run)
```

```
myrt_memcpy(dst, src, size, Direction::HostToDevice)
│
└─ mem_->memcpy(dst, src, size, Direction::HostToDevice)
```

---

## 통계 계산

```
divergence_rate  = masked_instructions / total_instructions

throughput_giops = total_instructions / elapsed_ns
                 = (total_instr × 10^-9) / (elapsed_ns × 10^-9)
                 = total_instr / elapsed_ns   [GIOPS]
```

`elapsed_ns`는 `std::chrono::high_resolution_clock`으로 측정.

---

## 구현 흐름

```
[1] gpurt/runtime.hpp 작성
    └─ KernelFunc 타입 alias
    └─ dim3 구조체
    └─ MyGPURuntime 클래스 선언

[2] gpurt/runtime.cpp 작성
    └─ 생성자: MemoryManager, WarpScheduler 초기화
    └─ myrt_malloc / myrt_free: mem_에 위임
    └─ myrt_memcpy: mem_에 위임
    └─ myrt_launch:
       └─ kernel(args) 호출 → Program 획득
       └─ grid·block 차원 순회하며 Block 생성
       └─ chrono 타이머 시작
       └─ scheduler_->run() 호출
       └─ 타이머 종료, 통계 누적
    └─ myrt_sync: print_stats() 호출, scheduler_ 리셋
    └─ print_stats: "[STATS] divergence: X.X%, throughput: X.X GIOPS" 출력

[3] test/unit/test_runtime.cpp 작성 (TDD)
    └─ TEST(Runtime, MallocFree)         — 할당·해제 후 재할당 가능 확인
    └─ TEST(Runtime, MemcpyRoundtrip)    — H→D→H 후 값 동일 확인
    └─ TEST(Runtime, LaunchSimpleKernel) — VADD 커널 실행 후 결과 검증
    └─ TEST(Runtime, DivergenceRate)     — BRA_DIV 포함 커널 실행 후 rate > 0
    └─ TEST(Runtime, ThroughputPositive) — throughput_giops() > 0.0
    └─ TEST(Runtime, SyncResetsStats)    — sync() 후 stats 초기화 확인
```

---

## 설계 결정 및 이유

**`KernelFunc = std::function<Program(void**)>`**
: 커널을 "명령어 시퀀스를 반환하는 함수"로 모델링. 실제 GPU 커널 컴파일과 유사하게, 런타임이 커널 코드(Program)를 받아 실행기(Scheduler)에 넘긴다.

**동기 실행 (no async)**
: 시뮬레이터 목적상 멀티스레딩 복잡도보다 실행 흐름의 명확성이 중요. `myrt_sync()`는 통계 출력 트리거 역할만 한다.

**`dim3 grid × block`으로 Block 생성**
: CUDA 런타임과 동일한 인터페이스. grid.x=4, block.x=128이면 512개 스레드를 16 Warp × 4 Block으로 실행.

**통계를 Runtime에서 집계 (Scheduler에서 누적 후 전달)**
: 여러 커널 실행에 걸친 누적 통계가 필요. Scheduler는 단일 커널 실행 통계만 알고, Runtime이 전체를 집계.

---

## 다음 단계

→ [06_ray_triangle_kernel.md](06_ray_triangle_kernel.md) — Ray-Triangle 커널 구현 및 PPM 출력

---

## 현재 코드·검증 대조

- 구현: [gpurt/runtime.hpp](../../gpurt/runtime.hpp), [gpurt/runtime.cpp](../../gpurt/runtime.cpp)
- 검증: [test/unit/test_runtime.cpp](../../test/unit/test_runtime.cpp)
- 측정이 있는 단계는 해당 벤치마크의 소스와 `test/benchmark/RESULTS.md`를 함께
  기준으로 삼는다. 이 문서의 수치는 실행 당시의 결과이며, 현재 재현값은 실행 환경에
  따라 달라질 수 있다.

---

[← 이전 문서](./04_warp_scheduler.md) · [구현 노트 목차](./README.md) · [다음 문서 →](./06_ray_triangle_kernel.md)
