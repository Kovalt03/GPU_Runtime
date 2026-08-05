cmake_minimum_required(VERSION 3.16)
project(gpu_runtime_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)   # -std=c++17, not gnu++17

# Default to Debug. Only meaningful for single-config generators.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# compile_commands.json — for clangd / IDE indexing
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warning settings live in one interface target so every target shares them.
# -Wswitch (part of -Wall) catches a missing case in opcode_name() at compile
# time whenever an opcode is added, so -Wall here is a safety net, not a style
# preference.
add_library(project_warnings INTERFACE)
target_compile_options(project_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# ---------------------------------------------------------------------------
# Core library — each implementation stage adds its source here.
#   1) ISA        src/isa.cpp        (done)
#   2) Memory     src/memory.cpp     (done)
#   3) Thread     src/thread.cpp     (done)
#   4) Scheduler  src/scheduler.cpp  (done)
#   5) Runtime    src/runtime.cpp
# ---------------------------------------------------------------------------
add_library(gpurt STATIC
    src/isa.cpp
    src/memory.cpp
    src/thread.cpp
    src/scheduler.cpp
)
target_include_directories(gpurt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gpurt PRIVATE project_warnings)

enable_testing()
add_subdirectory(tests)

# ---------------------------------------------------------------------------
# Formatting — `cmake --build build --target format` / `format-check`.
# Both delegate to scripts/format.sh so the rules live in exactly one place.
# ---------------------------------------------------------------------------
add_custom_target(format
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/format.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Applying .clang-format to all sources"
    VERBATIM
)

add_custom_target(format-check
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/format.sh --check
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking formatting"
    VERBATIM
)

# Added in later stages:
# add_subdirectory(kernels)
# add_subdirectory(benchmarks)
