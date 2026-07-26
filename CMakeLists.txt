cmake_minimum_required(VERSION 3.26)

project(spmv LANGUAGES CXX CUDA)

set(UCX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/ucx")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)

add_library(spmv_lib
    src/csr_matrix.cpp
    src/cuda_event_chrono.cu
    src/dense_vector.cpp
    src/metrics_mpi.cpp
    src/metrics.cpp
    src/mpi_common.cpp
    src/mtx_parser.cpp
    src/spmv_common.cu
    src/spmv_cusparse.cu
    src/spmv_oned_block_cudaaware.cu
    src/spmv_oned_block.cu
    src/spmv_oned_cyclic.cu
)

target_include_directories(spmv_lib PUBLIC
    include
    ${UCX_ROOT}/include
)

target_link_libraries(spmv_lib PUBLIC
    CUDA::cudart
    CUDA::cusparse
    MPI::MPI_CXX
    ${UCX_ROOT}/lib/libucp.so
    ${UCX_ROOT}/lib/libucs.so
    ${UCX_ROOT}/lib/libuct.so
)

target_compile_definitions(spmv_lib PRIVATE MPIX_CUDA_AWARE_SUPPORT)

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

# Non-MPI tests
add_executable(spmv_tests
    tests/test_spmv.cpp
)

target_link_libraries(spmv_tests PRIVATE spmv_lib gtest_main)

target_include_directories(spmv_tests PRIVATE tests)

add_test(NAME SpMVTests COMMAND spmv_tests)

set_tests_properties(SpMVTests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# MPI tests
add_executable(spmv_mpi_tests
    tests/test_spmv_mpi.cpp
)

target_link_libraries(spmv_mpi_tests PRIVATE spmv_lib gtest)

target_include_directories(spmv_mpi_tests PRIVATE tests)

add_test(NAME SpMVMpiTests
    COMMAND mpirun -np 2 $<TARGET_FILE:spmv_mpi_tests>
)

set_tests_properties(SpMVMpiTests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)