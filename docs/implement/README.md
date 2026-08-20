# 구현 노트

GPU Runtime Simulator를 실제로 구현하며 남긴 한글 설계·구현 보고서입니다. 문서는 기계를 만든 순서대로 이어지며, 각 단계에서 무엇을 만들었는지, 왜 그 경계를 택했는지, 어떤 검증과 측정을 했는지 기록합니다.

현재 코드와의 대조, 전체 테스트, 그리고 재현 가능한 벤치마크 명령은 이 구현 기록을 보완하는 [벤치마크](../benchmarks.md), [측정 결과](../findings.md), [모델 범위](../scope.md)에 있습니다.

## 공개 전 대조

- 2026-08-20 현재 소스 트리(`gpurt/`, `apps/`, `test/`)를 기준으로 각 보고서 끝에
  구현 파일과 검증 파일을 연결했다.
- `cmake --build build -j4`가 완료됐고, 병렬 CTest 실행에서 400개 테스트가 통과한 뒤
  남은 401–408번을 별도로 재실행하여 8/8 통과했다. 즉 408개 전체 테스트가 통과했다.
- 측정 수치는 해당 단계에서 기록한 결과다. 재실행할 때는 [RESULTS.md](../../test/benchmark/RESULTS.md)의
  환경과 명령을 기준으로 비교하며, 호스트 환경에 따라 wall-clock 값은 달라질 수 있다.

## 기반 계층

- [가상 기계 명세](./00_machine_spec.md)
- [Virtual ISA](./01_virtual_isa.md)
- [Memory Model](./02_memory_model.md)
- [Thread / Warp / ThreadBlock](./03_thread_warp.md)
- [Warp Scheduler](./04_warp_scheduler.md)
- [Runtime API](./05_runtime_api.md)
- [Ray-Triangle Intersection Kernel](./06_ray_triangle_kernel.md)
- [Warp Divergence 벤치마크](./07_divergence_benchmark.md)
- [IRBuilder](./08_kernel_builder.md)

## 렌더링 경로와 설계 기준

- [렌더링 파이프라인](./09_graphics_pipeline.md)
- [BARRIER와 공유 메모리](./10_barrier_and_shared_memory.md)
- [실제 하드웨어와의 간극](./11_hardware_gaps.md)

## 실행·메모리 시스템 확장

- [워프 스케줄링 가이드](./12_warp_scheduling_guide.md)
- [지연과 Coalescing 가이드](./13_latency_and_coalescing_guide.md)
- [지속 버퍼](./14_persistent_buffers.md)
- [early-Z](./15_early_z.md)
- [다중 SM](./16_multi_sm.md)
- [스트림 + indirect launch](./17_streams.md)
- [cp.async](./18_cp_async.md)
- [atomic](./19_atomics.md)

## 워크로드로 검증한 기능

- [행렬 유닛](./20_matrix_unit.md)
- [SER · 클러스터](./21_ser_and_clusters.md)
- [타일 행렬 곱](./22_gemm.md)
- [대역폭 / 큐잉](./23_bandwidth.md)
- [사용자 프래그먼트 셰이더 + varying](./24_shader.md)
- [BVH](./25_bvh.md)
- [인스턴싱](./26_instancing.md)
- [TLAS](./27_tlas.md)
- [재질 재정렬](./28_material_reorder.md)
- [GPU-driven](./29_gpu_driven.md)
- [스키닝 · 애니메이션](./30_skinning.md)

---

[← 프로젝트 README](../../README.md) · [벤치마크](../benchmarks.md) · [측정 결과](../findings.md)
