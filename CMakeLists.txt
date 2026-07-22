cmake_minimum_required(VERSION 3.26)

project(spmv LANGUAGES CXX CUDA)

# --- Local UCX Build ---
set(UCX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/ucx-install")
list(APPEND CMAKE_PREFIX_PATH "${UCX_ROOT}")
include_directories(${UCX_ROOT}/include)
link_directories(${UCX_ROOT}/lib)
# ------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)

add_library(spmv_lib
    src/csr_matrix.cpp
    src/cuda_event_chrono.cu
    src/dense_vector.cpp
    src/metrics.cpp
    src/mtx_parser.cpp
    src/spmv_common.cu
    src/spmv_csr_adaptive.cu
    src/spmv_csr_stream.cu
    src/spmv_csr_vector.cu
    src/spmv_cusparse.cu
)

target_include_directories(spmv_lib PUBLIC include)

target_link_libraries(spmv_lib
    PUBLIC
        CUDA::cudart
        CUDA::cusparse
        MPI::MPI_CXX
)

set_target_properties(spmv_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Main executable
add_executable(spmv src/main.cpp)

target_link_libraries(spmv PRIVATE spmv_lib)

set_target_properties(spmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

# GoogleTest
add_subdirectory(external/googletest-1.17.0)

enable_testing()

add_executable(spmv_tests
    tests/test_spmv.cpp
)

target_link_libraries(spmv_tests PRIVATE
    spmv_lib
    gtest_main
)

target_include_directories(spmv_tests PRIVATE include)

add_test(NAME SpMVTests COMMAND spmv_tests)

set_tests_properties(SpMVTests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)