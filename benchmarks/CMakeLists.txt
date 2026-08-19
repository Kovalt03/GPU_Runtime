# No per-target optimisation here, which is what these used to carry.
#
# It did nothing: every instruction these programs time is dispatched inside
# libgpurt, so -O2 on the benchmark's own translation unit optimised the loop
# that calls the library and left the library at -O0. A Debug build ran the
# mesh sweep in 41 seconds against 7 for Release.
#
# Seconds are therefore only worth reading from a Release build, which is what
# scripts/bench.sh makes. The lane and warp counts are the same either way.
add_executable(divergence_bench divergence_bench.cpp)
target_link_libraries(divergence_bench PRIVATE gpurt project_warnings)

add_executable(render_bench render_bench.cpp)
target_link_libraries(render_bench PRIVATE gpurt project_warnings)

add_executable(reduction_bench reduction_bench.cpp)
target_link_libraries(reduction_bench PRIVATE gpurt project_warnings)

add_executable(cache_bench cache_bench.cpp)
target_link_libraries(cache_bench PRIVATE gpurt project_warnings)

# The same comparisons render_bench makes, under the cost models it does not use.
add_executable(model_bench model_bench.cpp)
target_link_libraries(model_bench PRIVATE gpurt project_warnings)

# One kernel against several machines, where the others measure kernels against
# each other on one.
add_executable(occupancy_bench occupancy_bench.cpp)
target_link_libraries(occupancy_bench PRIVATE gpurt project_warnings)

# The first benchmark to hand the machine two kernels at once.
add_executable(stream_bench stream_bench.cpp)
target_link_libraries(stream_bench PRIVATE gpurt project_warnings)

# What a copy the warp does not wait for buys — on its own, and in a renderer.
add_executable(async_bench async_bench.cpp)
target_link_libraries(async_bench PRIVATE gpurt project_warnings)

# One instruction against 4,096 multiply-adds.
add_executable(mma_bench mma_bench.cpp)
target_link_libraries(mma_bench PRIVATE gpurt project_warnings)

# What regrouping a block's threads is worth, and when it is not.
add_executable(ser_bench ser_bench.cpp)
target_link_libraries(ser_bench PRIVATE gpurt project_warnings)

# What a block reading its neighbour's shared memory is worth.
add_executable(cluster_bench cluster_bench.cpp)
target_link_libraries(cluster_bench PRIVATE gpurt project_warnings)

# The kernel V_MMA_16X16X16_F32 and cp.async were both waiting for.
add_executable(gemm_bench gemm_bench.cpp)
target_link_libraries(gemm_bench PRIVATE gpurt project_warnings)

# What happens when the memory system has a ceiling.
add_executable(bandwidth_bench bandwidth_bench.cpp)
target_link_libraries(bandwidth_bench PRIVATE gpurt project_warnings)

# The last algorithm here that was still linear in the scene.
add_executable(bvh_bench bvh_bench.cpp)
target_link_libraries(bvh_bench PRIVATE gpurt project_warnings)
