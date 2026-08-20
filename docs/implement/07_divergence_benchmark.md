# 7단계 — Warp Divergence 벤치마크

## 목표

Warp divergence 비율을 인위적으로 조절하면서 throughput 변화를 측정한다.
"divergence가 증가할수록 throughput이 감소한다"는 GPU 아키텍처의 핵심 원리를
수치로 증명하여 포트폴리오의 기술적 깊이를 보여준다.

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `test/benchmark/src/divergence_bench.cpp` | 벤치마크 시나리오 정의 및 결과 출력 |

---

## 벤치마크 시나리오

divergence 비율을 0%~100% 사이에서 단계적으로 변화시킨다.

```
divergence 0%   → 모든 스레드가 동일 경로 (BRA_DIV 없음)
divergence 25%  → Warp 내 8개 lane이 분기, 24개는 직진
divergence 50%  → 16개 vs 16개
divergence 75%  → 24개 분기, 8개 직진
divergence 100% → 모든 lane이 서로 다른 경로 (최악)
```

### 제어 방식

각 시나리오에서 커널을 다르게 구성한다:

```
// divergence_level = 0..32 (분기하는 lane 수)
// 시드 고정으로 재현 가능: 레지스터 초기값을 결정론적으로 설정

Thread i의 초기 조건 레지스터:
  reg[cond] = (i < divergence_level) ? 1.0f : 0.0f
  → BRA_DIV 실행 시 앞쪽 lane들만 분기
```

---

## 측정 항목

| 항목 | 단위 | 설명 |
|------|------|------|
| divergence_rate | % | masked_instructions / total_instructions |
| throughput | GIOPS | effective instructions / elapsed time |
| slowdown | 배율 | divergence=0% 대비 상대 성능 감소 |

---

## 예상 출력

```
[BENCH] Warp Divergence vs Throughput
---------------------------------------------
 Divergence |  Throughput | Slowdown
------------|-------------|----------
      0.0%  |  X.XX GIOPS |  1.00x
     12.5%  |  X.XX GIOPS |  X.XXx
     25.0%  |  X.XX GIOPS |  X.XXx
     37.5%  |  X.XX GIOPS |  X.XXx
     50.0%  |  X.XX GIOPS |  X.XXx
     62.5%  |  X.XX GIOPS |  X.XXx
     75.0%  |  X.XX GIOPS |  X.XXx
     87.5%  |  X.XX GIOPS |  X.XXx
    100.0%  |  X.XX GIOPS |  X.XXx
---------------------------------------------
[BENCH] Peak: X.XX GIOPS (0% divergence)
[BENCH] Min:  X.XX GIOPS (100% divergence)
```

---

## 구현 흐름

```
[1] 벤치마크 커널 팩토리 작성
    └─ make_divergence_kernel(uint32_t diverging_lanes) → KernelFunc
       └─ diverging_lanes개 lane: BRA_DIV → 별도 경로 실행 후 BRA로 합류
       └─ 나머지 lane: 직선 경로

[2] 시나리오 실행 루프
    └─ for diverging in {0, 4, 8, 12, 16, 20, 24, 28, 32}:
        ├─ MyGPURuntime rt
        ├─ myrt_launch(make_divergence_kernel(diverging), grid, block, args)
        ├─ myrt_sync()
        └─ 결과 수집: divergence_rate(), throughput_giops()

[3] 결과 출력
    └─ 표 형식으로 stdout 출력
    └─ (선택) CSV 파일 저장 → Python/gnuplot으로 그래프 생성 가능

[4] 재현성 보장
    └─ 레지스터 초기값 고정 (시드 없이 결정론적 할당)
    └─ Warp 수 고정: 1024 Warp (32768 threads)
    └─ 커널 반복 횟수 고정: 1회 (시뮬레이터는 결정론적)
```

---

## 재현성 조건

| 항목 | 고정값 |
|------|--------|
| Warp 수 | 1024 |
| Thread/Warp | 32 |
| 총 스레드 수 | 32768 |
| 레지스터 초기값 | 결정론적 (시드 없음) |
| 컴파일 옵션 | `-O2 -std=c++17` |

---

## 포트폴리오 활용

README.md에 포함할 그래프:
```
Throughput (GIOPS)
│
X.X ┤■■■■
    │    ■■
X.X ┤      ■■■
    │         ■■
X.X ┤            ■■■
    │                ■■■■■
X.X ┤                     ■■■■■■■■
    └──────────────────────────────── Divergence (%)
      0%   25%   50%   75%  100%
```

"divergence 100%에서 throughput이 N배 감소"라는 수치를 README에 기재.

---

## 설계 결정 및 이유

**결정론적 레지스터 초기값 (시드 고정 없음)**
: 시뮬레이터는 CPU 위에서 실행되므로 실행 환경에 무관하게 동일 결과. 무작위 요소 없이 diverging_lanes 수만으로 제어.

**1024 Warp 기준**
: 측정 시간이 너무 짧으면 타이머 오차가 크고, 너무 길면 반복 실험 불편. 1024 Warp × 1000 instructions ≈ 10^7 ops — 적당한 규모.

**CSV 출력 옵션**
: 그래프 생성을 위해 `--csv` 플래그로 `output/divergence_bench.csv` 저장. Python matplotlib으로 시각화하여 README에 삽입.

---

## 다음 단계

→ [09_graphics_pipeline.md](09_graphics_pipeline.md) — World → View → Projection
파이프라인으로 여러 삼각형 렌더링. 이 문서의 divergence 측정이 그쪽
"분기 vs 프리디케이션" 비교의 기준선이 된다.

---

## 현재 코드·검증 대조

- 구현: [test/benchmark/src/divergence_bench.cpp](../../test/benchmark/src/divergence_bench.cpp)
- 검증: [test/unit/test_scheduler.cpp](../../test/unit/test_scheduler.cpp)
- 측정이 있는 단계는 해당 벤치마크의 소스와 `test/benchmark/RESULTS.md`를 함께
  기준으로 삼는다. 이 문서의 수치는 실행 당시의 결과이며, 현재 재현값은 실행 환경에
  따라 달라질 수 있다.

---

[← 이전 문서](./06_ray_triangle_kernel.md) · [구현 노트 목차](./README.md) · [다음 문서 →](./08_kernel_builder.md)
