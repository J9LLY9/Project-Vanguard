cmake_minimum_required(VERSION 3.18)
project(Vanguard LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CPU_SRCS
    src/core/model/gpt2_weights.cpp
    src/core/model/gpt2_tokenizer.cpp
    src/backends/cpu/cpu_kernels.cpp
    src/backends/cpu/cpu_engine.cpp
)

set(RUNTIME_SRCS
    src/runtime/hardware_probe.cpp
)

# CPU Correctness Test
add_executable(test_cpu_correctness
    tests/integration/test_cpu_correctness.cpp
    ${CPU_SRCS}
)
set_target_properties(test_cpu_correctness PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)

# CUDA Support Check
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(CUDA_SRCS
        src/backends/cuda/kv_cache.cu
        src/backends/cuda/cuda_engine.cu
    )

    add_executable(test_cuda_correctness
        tests/integration/test_cuda_correctness.cpp
        ${CPU_SRCS}
        ${CUDA_SRCS}
        ${RUNTIME_SRCS}
    )
    set_target_properties(test_cuda_correctness PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
    )

    add_executable(vanguard_cli
        apps/cli/main.cpp
        ${CPU_SRCS}
        ${CUDA_SRCS}
        ${RUNTIME_SRCS}
    )
    target_compile_definitions(vanguard_cli PRIVATE VANGUARD_WITH_CUDA)
    set_target_properties(vanguard_cli PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
    )
else()
    message(STATUS "CUDA compiler not found. Building CPU-only target for CLI.")
    add_executable(vanguard_cli
        apps/cli/main.cpp
        ${CPU_SRCS}
    )
    set_target_properties(vanguard_cli PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
    )
endif()
