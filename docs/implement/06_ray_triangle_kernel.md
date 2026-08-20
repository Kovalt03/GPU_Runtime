# 6단계 — Ray-Triangle Intersection Kernel

## 목표

Virtual ISA 명령어(`Program`)로 Möller–Trumbore Ray-Triangle intersection을 구현하고,
256×256 해상도로 렌더링한 결과를 PPM 파일로 저장한다.
이 커널이 전체 스택의 통합 검증 워크로드이다.

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `apps/ray_triangle.cpp` | 커널 정의 + main() (PPM 출력) |
| `output/result.ppm` | 렌더링 결과 (런타임 생성) |

---

## Möller–Trumbore 알고리즘

```
입력:  Ray origin O, direction D
       Triangle vertices V0, V1, V2

E1 = V1 - V0
E2 = V2 - V0
h  = cross(D, E2)
a  = dot(E1, h)          // ≈ 0 이면 평행 (miss)

f  = 1.0 / a
s  = O - V0
u  = f * dot(s, h)       // u < 0 또는 u > 1 → miss

q  = cross(s, E1)
v  = f * dot(D, q)       // v < 0 또는 u+v > 1 → miss

t  = f * dot(E2, q)      // t > 0 이면 hit

출력:  hit 여부, t (교차 거리)
```

---

## ISA 매핑 전략

```
고수준 수식 → Virtual ISA 명령어

E1 = V1 - V0:
  regs[0..2] = V0 (LD_GLOBAL × 3)
  regs[3..5] = V1 (LD_GLOBAL × 3)
  VADD_F32 r6, r3, -r0   (VADD + 음수 imm 트릭 or 별도 VSUB 정의 고려)
  ...

cross(A, B) → VMUL × 3 + VADD × 3 (6 instructions)
dot(A, B)   → VDOT3 (1 instruction)
normalize   → VNORM3 (1 instruction)

픽셀당 분기 (hit/miss):
  BRA_DIV cond_reg, miss_offset  ← divergence 발생 지점
  (화면 내 삼각형 경계에서 Warp 내 일부 lane은 hit, 나머지는 miss)
```

---

## 커널 구조

```
ray_triangle_kernel(void** args):
  args[0] = float* framebuffer  (device, 256×256×3 floats)
  args[1] = float* vertices     (device, 9 floats: V0, V1, V2)
  args[2] = float* camera       (device, 6 floats: origin + direction_base)
  args[3] = int*   resolution   (device, 2 ints: width, height)

  → Program (ISA 명령어 시퀀스 반환)

각 스레드 = 1개 픽셀
  thread_id = warp_id * 32 + lane_id
  px = thread_id % width
  py = thread_id / width

  ray_dir = normalize(vec3(px/width - 0.5, py/height - 0.5, -1.0))
  → Möller–Trumbore 실행
  → hit: framebuffer[px,py] = (u, v, 1-u-v)  (무게중심 좌표 → 색상)
  → miss: framebuffer[px,py] = (0, 0, 0)      (배경 = 검정)
```

---

## PPM 출력 포맷

```
P3
256 256
255
r g b r g b r g b ...   ← 각 픽셀을 0~255 정수로 출력
```

```cpp
// main() 에서 D→H 복사 후 PPM 작성
std::ofstream f("output/result.ppm");
f << "P3\n" << width << " " << height << "\n255\n";
for (int i = 0; i < width * height; ++i) {
    int r = static_cast<int>(std::clamp(fb[i*3+0], 0.0f, 1.0f) * 255);
    int g = static_cast<int>(std::clamp(fb[i*3+1], 0.0f, 1.0f) * 255);
    int b = static_cast<int>(std::clamp(fb[i*3+2], 0.0f, 1.0f) * 255);
    f << r << " " << g << " " << b << "\n";
}
```

---

## 구현 흐름

```
[1] 씬 설정 (main 상단)
    └─ Triangle vertices: V0(0,-0.5,−2), V1(−0.5,0.5,−2), V2(0.5,0.5,−2)
    └─ Camera origin: (0, 0, 0), FOV를 통한 ray direction 계산

[2] Host 데이터 준비 및 D→H 전송
    └─ myrt_malloc(framebuffer, vertices, camera, resolution)
    └─ myrt_memcpy(HostToDevice) for vertices, camera, resolution

[3] 커널 함수 정의 (KernelFunc)
    └─ args를 레지스터에 로드 (LD_GLOBAL)
    └─ thread_id → (px, py) 계산
    └─ ray direction 계산 (VMUL, VADD, VNORM3)
    └─ Möller–Trumbore: E1, E2, h, a, s, u, q, v, t (VDOT3, VMUL, VADD 조합)
    └─ BRA_DIV → hit/miss 분기
    └─ 프레임버퍼 쓰기 (ST_GLOBAL × 3)
    └─ RET

[4] 커널 실행
    └─ myrt_launch(kernel, grid, block={32,1,1}, args)
    └─ myrt_sync()  → [STATS] 출력

[5] 결과 D→H 복사 및 PPM 작성
    └─ myrt_memcpy(DeviceToHost)
    └─ output/result.ppm 파일 작성

[6] 시각적 검증
    └─ PPM을 열어 삼각형이 화면 중앙에 보이는지 확인
    └─ 삼각형 영역: 무게중심 좌표 색상 (RGB 그라디언트)
    └─ 배경: 검정
```

---

## 예상 렌더링 결과

```
+--------------------+
| (black background) |
|                    |
|       /\           | <- triangle: barycentric colour
|      /  \          |    vertices: R, G, B
|     /    \         |
|    /______\        |
|                    |
+--------------------+
```

Warp divergence가 가장 많이 발생하는 지점: 삼각형 **경계 픽셀**
(같은 Warp 내 일부 lane은 hit, 나머지는 miss → BRA_DIV에서 diverge)

---

## 설계 결정 및 이유

**무게중심 좌표 → 색상**
: 시각적으로 올바른지 즉시 확인 가능. hit한 경우 u, v, 1-u-v가 [0,1] 범위 → RGB로 자연스럽게 매핑.

**32 threads = 1 Warp = 32 픽셀 처리**
: 스레드 1개 = 픽셀 1개. 256×256 = 65536 픽셀 → 2048 Warp.

**삼각형을 화면 중앙에 배치**
: z=-2, FOV ~60° 기준으로 적당한 크기의 삼각형이 보임. Warp divergence 발생을 위해 삼각형 경계가 여러 Warp를 가로질러야 함.

---

## 다음 단계

→ [07_divergence_benchmark.md](07_divergence_benchmark.md) — Divergence 성능 측정

---

## 현재 코드·검증 대조

- 구현: [apps/ray_triangle.cpp](../../apps/ray_triangle.cpp), [gpurt/isa.hpp](../../gpurt/isa.hpp)
- 검증: [test/unit/test_pipeline.cpp](../../test/unit/test_pipeline.cpp)
- 측정이 있는 단계는 해당 벤치마크의 소스와 `test/benchmark/RESULTS.md`를 함께
  기준으로 삼는다. 이 문서의 수치는 실행 당시의 결과이며, 현재 재현값은 실행 환경에
  따라 달라질 수 있다.

---

[← 이전 문서](./05_runtime_api.md) · [구현 노트 목차](./README.md) · [다음 문서 →](./07_divergence_benchmark.md)
