# 2단계 — Memory Model

## 목표

Host(CPU)와 Device(GPU)의 메모리를 완전히 분리된 주소 공간으로 시뮬레이션한다.
Free-list 방식으로 동적 할당/해제를 관리하고, 명시적 전송 API로 데이터를 이동한다.

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `gpurt/memory.hpp` | MemoryManager 클래스, Direction enum, Block 구조체 선언 |
| `gpurt/memory.cpp` | 할당(alloc), 해제(free), 전송(memcpy) 구현 |
| `test/unit/test_memory.cpp` | GoogleTest — 할당·해제·전송·경계 조건 검증 |

---

## 주소 공간 분리 원칙

```
Host Memory (hostMem)           Device Memory (deviceMem)
+--------------------+          +--------------------+
|  std::vector<u8>   |  memcpy  |  std::vector<u8>   |
|  CPU directly uses | <------> |  GPU kernel uses   |
+--------------------+          +--------------------+
          ^                               ^
   void* (host ptr)              void* (device ptr)
   hostMem.data() + offset       deviceMem.data() + offset
```

> **규칙**: `deviceMem`의 포인터로 `hostMem`에 접근하거나 그 반대를 시도하면
> `std::runtime_error`를 던진다. 포인터 범위를 검사해 혼용을 탐지한다.

---

## 데이터 구조

실제 인터페이스는 `gpurt/memory.hpp` 참조. 요약하면:

```cpp
enum class Direction { HostToDevice, DeviceToHost };

struct Block {
    size_t offset;  // 버퍼 내 바이트 오프셋
    size_t size;    // 블록 크기 (바이트)
    bool   free;    // 현재 비어 있는지 여부
};

class MemoryManager {
public:
    static constexpr size_t ALLOC_ALIGNMENT = 16;

    explicit MemoryManager(size_t host_size, size_t device_size);
    MemoryManager(const MemoryManager&) = delete;   // 나눠준 포인터가 이 인스턴스 소유

    void* host_alloc(size_t size);      void host_free(void* ptr);
    void* device_alloc(size_t size);    void device_free(void* ptr);

    void memcpy(void* dst, const void* src, size_t size, Direction dir);

    size_t host_free_bytes() const;
    size_t device_free_bytes() const;
    size_t device_largest_free_block() const;   // 단편화 관측용
    bool   is_host_ptr(const void* ptr) const;
    bool   is_device_ptr(const void* ptr) const;

private:
    // host/device는 "어느 버퍼를 쪼개느냐"만 다르므로 할당 로직을 한 벌만 둔다.
    struct Arena {
        std::vector<uint8_t> bytes;
        std::list<Block>     blocks;   // used + free 전부. 이웃을 봐야 병합 가능
    };
    Arena host_;
    Arena device_;

    static void*  alloc_from(Arena&, size_t size);
    static void   free_from(Arena&, void* ptr, const char* space_name);
    static void   coalesce(Arena&);
    static size_t free_bytes(const Arena&);
    static bool   contains(const Arena&, const void* ptr);
    static void   require_in_range(const Arena&, const void*, size_t, const char*);
};
```

> **`blocks`는 free 블록만 담는 리스트가 아니다.** used 블록도 함께, 오프셋
> 오름차순으로 빈틈없이 들어간다. 해제된 블록의 양옆이 비었는지 알아야
> 병합할 수 있기 때문이다.

---

## Free-list 할당 전략

```
초기 상태:  [ FREE  |  전체 device_size  ]

alloc(100):
  first-fit 탐색 → 앞에서부터 100바이트 이상 free 블록 찾음
  → [ USED(100) | FREE(나머지) ]

alloc(50):
  → [ USED(100) | USED(50) | FREE(나머지) ]

free(ptr_to_100):
  → [ FREE(100) | USED(50) | FREE(나머지) ]

coalesce():
  인접 free 블록 병합 시도 → 여기서는 100과 나머지가 인접하지 않으므로 유지
  → [ FREE(100) | USED(50) | FREE(나머지) ]

free(ptr_to_50):
  → [ FREE(100) | FREE(50) | FREE(나머지) ]
  coalesce() → [ FREE(전체) ]
```

**first-fit 선택 이유**: best-fit 대비 구현이 단순하고, 시뮬레이터 규모에서 단편화 차이가 미미하다.

### 불변식: 인접한 free 블록 2개는 (거의) 존재하지 않는다

`coalesce`가 끝난 시점에는 **어떤 두 이웃 블록도 동시에 free가 아니다.** `alloc`은
`[used | free]`만 만들고, `free`는 항상 `coalesce`로 끝나기 때문이다.

따라서 연속 free 블록은 **해제 직후 일시적으로만** 생기고, 최대 3개다:

```
해제 전 (불변식 성립)   [ FREE100 | USED50 | FREE3946 ]
                                    ↑ 이걸 해제
해제 직후 (일시적)      [ FREE100 | FREE50 | FREE3946 ]   ← 3연속
coalesce 후             [ FREE4096 ]
```

4개 이상은 불가능하다. `FREE100`의 왼쪽 블록이 free였다면 해제 전에 이미
불변식이 깨져 있었을 것이기 때문이다.

이것이 **이웃만 병합해도 충분한 이유**이자, 전체 훑기를 쓸 때 **병합 후
반복자를 전진시키면 안 되는 이유**다 (`FREE100+FREE50` 병합 후 결과를
`FREE3946`과 다시 비교해야 한다).

---

## 구현 순서

`gpurt/memory.hpp`와 `test/unit/test_memory.cpp`(17개)는 이미 있다.
`gpurt/memory.cpp`를 아래 순서로 채운다.

**순서의 원칙: 매 단계마다 통과하는 테스트가 늘어나도록 배치했다.**
20~30줄 쓸 때마다 `ctest`가 답을 주므로, 마지막에 몰아서 디버깅할 일이 없다.
`test/unit/CMakeLists.txt`의 `add_layer_test(test_memory)` 주석을 먼저 풀고 시작한다.

---

### [1] 생성자 + `free_bytes` + `contains` / `is_*_ptr`

가장 먼저 하는 이유: **관측 수단이 없으면 이후 단계를 검증할 수 없다.**
할당기를 먼저 짜면 그게 맞는지 확인할 방법이 없다.

```
생성자   각 Arena: bytes.resize(size)
                 blocks = { Block{0, size, true} }   ← 전체가 하나의 free 블록
contains  ptr이 [bytes.data(), bytes.data() + size) 안에 있는가
free_bytes  blocks 순회하며 free인 것의 size 합
```

- `size == 0`인 Arena도 만들어질 수 있으니 `blocks`를 비워둘지 결정할 것
- **`bytes`는 여기서 딱 한 번만 `resize`한다.** 나중에 다시 resize하면
  이미 나눠준 포인터가 전부 dangling이 된다 (헤더 주석에도 명시)

→ `HostAndDeviceAreDistinctSpaces`의 `is_*_ptr` 부분이 통과하기 시작

---

### [2] `alloc_from` — first-fit + 분할

```
size를 ALLOC_ALIGNMENT 배수로 올림          ← 먼저 한다
blocks를 앞에서부터 순회
  free이고 size 이상인 첫 블록을 찾으면:
    남는 공간이 충분히 크면 → 뒤쪽을 새 free 블록으로 잘라 삽입
    남는 공간이 자투리면    → 자르지 않고 통째로 준다
    free = false 표시
    return bytes.data() + offset
못 찾으면 throw std::runtime_error
```

주의할 점 두 가지:

- **정렬 올림을 탐색 전에 한다.** 나중에 하면 "찾았는데 실제로는 안 들어가는"
  경우가 생긴다.
- **자투리 분할 금지.** 남는 공간이 `ALLOC_ALIGNMENT`보다 작으면 쪼개봐야
  아무도 못 쓰는 블록만 늘어난다. 통째로 주는 편이 낫다 (internal
  fragmentation을 감수하고 external fragmentation을 막는 쪽).
- 예외를 던질 때는 **`blocks`를 건드리기 전**이어야 한다. 실패한 할당이
  상태를 반쯤 바꿔놓으면 안 된다.

→ `AllocReturnsNonNull`, `AllocationsDoNotOverlap`, `AllocationsAreAligned`,
  `ThrowOnZeroSize`, `ThrowOnOOM` 통과

---

### [3] `free_from` — 검증 후 표시

```
ptr == nullptr → 그냥 return            (C의 free(nullptr)과 동일하게)
contains 아니면 → throw                 (다른 주소 공간 / 스택 변수)
offset = (uint8_t*)ptr - bytes.data()
blocks에서 offset이 정확히 일치하는 블록을 찾는다
  없으면 → throw                        (할당 중간을 가리키는 포인터)
  이미 free면 → throw                   (double free)
free = true
coalesce()
```

**"offset이 정확히 일치"가 핵심이다.** 블록 *안*에 들어가는지가 아니라
**시작점인지**를 봐야 `free(p + 16)` 같은 걸 잡아낸다. 이게 `FreeUnknownPointerThrows`,
`DoubleFreeThrows`가 검사하는 부분.

`space_name` 인자는 예외 메시지를 `"device_free: ..."`처럼 만들기 위한 것.
host/device 중 어느 쪽에서 터졌는지 메시지만 보고 알 수 있어야 한다.

→ `FreeAndReallocSameSize`, `FreeNullptrIsNoop`, `DoubleFreeThrows`,
  `FreeUnknownPointerThrows`, `HostAndDeviceAreDistinctSpaces` 통과

---

### [4] `coalesce` — 인접 free 병합

여기가 이 레이어에서 **버그가 가장 잘 나는 곳**이다.

```
it = blocks.begin()
while (it != blocks.end()):
    next = it 다음
    if (next 존재 && it->free && next->free):
        it->size += next->size
        blocks.erase(next)      ← it은 그대로 두고 계속 (3개 연속 병합)
    else:
        ++it
```

- **병합했으면 `it`을 전진시키지 않는다.** 전진시키면 free 3개가 연속일 때
  2개만 합쳐지고 끝난다. `CoalesceAfterFree`가 정확히 이 케이스를 잡는다.
- `blocks`가 오프셋 오름차순으로 유지된다는 전제 위에 서 있다. [2]의 분할이
  이 순서를 깨지 않도록 항상 **뒤쪽에** 삽입할 것.
- 오프셋은 건드리지 않는다 — 앞 블록의 offset이 그대로 병합 블록의 offset이다.

→ `CoalesceAfterFree`, `FullCycleRestoresCapacity` 통과

---

### [5] `memcpy` — 방향 검증 후 복사

```
size == 0 → 그냥 return할지 throw할지 정할 것
dir에 따라:
  HostToDevice : dst는 device여야 하고 [dst, dst+size)가 범위 안,
                 src는 device가 아니어야 함
  DeviceToHost : src는 device여야 하고 [src, src+size)가 범위 안,
                 dst는 device가 아니어야 함
std::memcpy(dst, src, size)
```

- **`std::memcpy`로 반드시 한정한다.** 이 멤버 함수 이름이 `memcpy`라
  한정하지 않으면 자기 자신을 호출해 무한 재귀에 빠진다.
- host 쪽은 `host_alloc`에서 온 포인터가 아니어도 된다 (설계 결정 참조).
  그래서 host 쪽 검사는 "device가 **아닐** 것" 하나뿐이다.
- 범위 검사는 시작 포인터만이 아니라 **`ptr + size`까지** 봐야 한다.
  `MemcpyRejectsOutOfBounds`가 이걸 잡는다.

→ 나머지 memcpy 테스트 4개 통과. `ctest` 17 → 34개

---

### 참고: 포인터 ↔ 오프셋 변환

```cpp
// offset → pointer
arena.bytes.data() + offset

// pointer → offset (const 유지)
static_cast<const uint8_t*>(ptr) - arena.bytes.data()
```

`void*`는 산술이 안 되므로 반드시 `uint8_t*`로 캐스팅한 뒤 계산한다.

---

## 설계 결정 및 이유

**`std::list<Block>` free-list**
: 블록 분할·병합 시 중간 삽입/삭제가 필요하므로 `std::vector` 대비 `std::list`가 적합. 시뮬레이터 규모에서 캐시 미스 패널티는 무시 가능.

**coalesce 즉시 호출**
: `free()` 즉시 병합하면 단편화를 조기에 해소. 지연 병합(lazy coalesce)은 구현 복잡도 대비 이점 없음.

**포인터 = `device_mem_.data() + offset`**
: 별도 핸들 테이블 없이 원시 포인터를 그대로 반환. 오프셋 역산이 O(n)이지만 할당 횟수가 적어 허용.

**주소 공간 혼용 탐지**
: `memcpy` 시 dst/src 포인터가 각각 어느 버퍼 범위에 속하는지 검사. 범위 외 포인터는 예외 처리.

**`host_alloc` / `host_free` 추가 (초안에서 변경)**
: 초안에는 `host_mem_` 멤버만 있고 거기서 할당하는 함수가 없어 죽은 멤버였다. device와 대칭으로 맞췄다. CUDA의 pinned memory에 해당하는 위치.

**`memcpy`의 host 쪽은 임의 포인터 허용 (초안에서 변경)**
: 커널은 `mm.memcpy(dev, framebuffer.data(), ...)`처럼 평범한 `std::vector`에서 복사한다. host 포인터를 `host_alloc` 출신으로 강제하면 쓰기 불편해지고, `cudaMemcpy`도 스택 주소를 그대로 받는다. 따라서 **device 쪽만 엄격히 검사**하고(범위 + 크기), host 쪽은 "device 범위가 아닐 것"만 확인한다. 문서 상단의 혼용 탐지 규칙은 이 검사만으로 충족된다.

**private 멤버를 `Arena`로 묶음 (초안에서 변경)**
: host에도 할당을 붙이면 first-fit·분할·병합 로직이 두 벌이 된다. `{버퍼 + 블록 리스트}`를 `Arena`로 묶고 `alloc_from(arena, size)` 형태의 static 헬퍼로 한 번만 구현한다.

**`ALLOC_ALIGNMENT = 16`**
: ISA가 float / vec3 단위로 로드하므로 최소한 자연 정렬은 보장해야 한다. 실제 GPU는 훨씬 거칠게(NVIDIA 256B) 잡지만, 4KB 아레나에서 256B 정렬은 낭비가 크다.

---

## 성능 측정 (2026-08-03)

**결론부터: 알고 안 고친 것이다.** 워크로드가 수십 회 할당이라 문제되지 않아
단순함을 택했다. README "기술적 의사결정 이유"에 쓸 근거.

측정 환경: Apple M-series (arm64), AppleClang 16, `-O2`, 64MB 아레나,
N회 64바이트 할당 후 전부 해제, 3회 반복 중 최솟값.

```
      N | 전체훑기 coalesce  | 이웃만 병합 + 오프셋 색인
        |      alloc    free |      alloc      free
   1000 |      0.80     0.69 |      0.86       0.05
   2000 |      2.36     2.42 |      2.41       0.05
   4000 |     13.58    11.36 |     12.46       0.10
   8000 |     57.75    62.23 |     91.97       0.21
```
(단위 ms. 색인 버전의 alloc이 느린 것은 `unordered_map` 삽입 비용이며
알고리즘 자체는 동일하다.)

### 관측 1 — `coalesce` 전체 훑기가 free를 O(n²)로 만든다

free 열: **62.23ms vs 0.21ms, 약 300배.** 해제 시 새로 합쳐질 수 있는 것은
그 블록과 좌우 이웃뿐인데(위 불변식) 매번 리스트 전체를 훑기 때문이다.

O(1) 버전:

```cpp
void coalesce_at(Arena& arena, It it) {
    auto next = std::next(it);
    if (next != arena.blocks.end() && next->free) {
        it->size += next->size;
        arena.blocks.erase(next);
    }
    if (it != arena.blocks.begin()) {
        auto prev = std::prev(it);
        if (prev->free) {
            prev->size += it->size;
            arena.blocks.erase(it);
        }
    }
}
```
뒤를 먼저 합치고 앞을 합쳐야 한다. 앞을 먼저 합치면 `it`이 무효화되어
뒤를 볼 수 없다.

### 관측 2 — 그래도 전체는 여전히 O(n²)

alloc 열은 두 방식이 같다. first-fit이 매번 앞에서부터 훑는데 앞쪽은 이미
`used` 블록으로 가득 차 있어 건너뛰기만 한다. `coalesce`만 고쳐도 전체
시간은 절반 정도밖에 줄지 않는다.

제대로 고치려면:
- **free 블록만 연결한 별도 리스트** — `used`를 건너뛰는 낭비 제거
- **크기별 버킷(segregated free list)** — 실제 `malloc` 구현들의 방식

### 관측 3 — 반복자 순회 형태는 무관하다

`while (std::next(it) != end())`와 `while (it != end())` + 본문 검사를
`-O2`로 컴파일하면 명령어 34개 vs 35개다. `std::next` 중복 호출은 컴파일러가
정리한다. 이 층위에서 성능을 논거로 삼을 수 없다.

---

## 다음 단계

→ [03_thread_warp.md](03_thread_warp.md) — Thread / Warp / Block 구조체

---

## 현재 코드·검증 대조

- 구현: [gpurt/memory.hpp](../../gpurt/memory.hpp), [gpurt/memory.cpp](../../gpurt/memory.cpp)
- 검증: [test/unit/test_memory.cpp](../../test/unit/test_memory.cpp)
- 측정이 있는 단계는 해당 벤치마크의 소스와 `test/benchmark/RESULTS.md`를 함께
  기준으로 삼는다. 이 문서의 수치는 실행 당시의 결과이며, 현재 재현값은 실행 환경에
  따라 달라질 수 있다.

---

[← 이전 문서](./01_virtual_isa.md) · [구현 노트 목차](./README.md) · [다음 문서 →](./03_thread_warp.md)
