# Built with optimisations even in a Debug configuration: the timings measure
# this simulator running on the host, and an unoptimised build would report the
# debug build's speed rather than anything about divergence.
add_executable(divergence_bench divergence_bench.cpp)
target_link_libraries(divergence_bench PRIVATE gpurt project_warnings)
target_compile_options(divergence_bench PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2>
)

add_executable(render_bench render_bench.cpp)
target_link_libraries(render_bench PRIVATE gpurt project_warnings)
target_compile_options(render_bench PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2>
)
