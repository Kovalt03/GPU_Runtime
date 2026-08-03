include(FetchContent)

# Fetch GoogleTest from source and build it alongside the project. No system
# gtest install is required, and pinning the tag keeps the benchmark environment
# reproducible.
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Register one test executable per layer.
# Each target lands at build/tests/<name>.
function(add_layer_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE gpurt project_warnings GTest::gtest_main)
    # Register each TEST() as its own ctest case, so a failure names the test.
    gtest_discover_tests(${name})
endfunction()

add_layer_test(test_isa)
add_layer_test(test_memory)
# add_layer_test(test_scheduler)
# add_layer_test(test_runtime)
