#include <gtest/gtest.h>
#include <mpi.h>

#include "csr_matrix.hpp"
#include "spmv_oned_block.cuh"
#include "spmv_oned_cyclic.cuh"
#include "spmv_oned_block_cudaaware.cuh"
#include "test_utils.hpp"

#include <filesystem>
#include <vector>
#include <string>

// This code was created with the help of generative artificial intelligence.

class SpMVMpiTest : public ::testing::TestWithParam<std::string>
{
protected:
    int rank;
    int world_size;
    std::string filename;
    MtxParser::MtxMatrix mtx_matrix;
    DenseVector x;
    DenseVector y;

    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        filename = GetParam();

        if (rank == 0) {
            mtx_matrix = MtxParser::parseMtxFile(filename);
            x = DenseVector::random_vector(mtx_matrix.cols);
        }

        y = DenseVector(mtx_matrix.rows);
    }
};

TEST_P(SpMVMpiTest, OneDBlockMatchesCPU) {
    DenseVector global_x = x;
    DenseVector global_y(mtx_matrix.rows);
    MetricsMpi metrics_mpi(2);

    spmv_oned_block(mtx_matrix, global_x, global_y, metrics_mpi);

    if (rank == 0) {
        CsrMatrix A(mtx_matrix);
        DenseVector y_cpu = A * x;
        EXPECT_TRUE(y_cpu.is_close(global_y)) << "1D block mismatch for file: " << filename;
    }
}

TEST_P(SpMVMpiTest, OneDCyclicMatchesCPU) {
    DenseVector global_x = x;
    DenseVector global_y(mtx_matrix.rows);
    MetricsMpi metrics_mpi(2);

    spmv_oned_cyclic(mtx_matrix, global_x, global_y, metrics_mpi);

    if (rank == 0) {
        CsrMatrix A(mtx_matrix);
        DenseVector y_cpu = A * x;
        EXPECT_TRUE(y_cpu.is_close(global_y)) << "1D cyclic mismatch for file: " << filename;
    }
}

TEST_P(SpMVMpiTest, OneDBlockCudaAwareMatchesCPU) {
    DenseVector global_x = x;
    DenseVector global_y(mtx_matrix.rows);
    MetricsMpi metrics_mpi(2);

    spmv_oned_block_cudaaware(mtx_matrix, global_x, global_y, metrics_mpi);

    if (rank == 0) {
        CsrMatrix A(mtx_matrix);
        DenseVector y_cpu = A * x;
        EXPECT_TRUE(y_cpu.is_close(global_y)) << "1D block CUDA aware mismatch for file: " << filename;
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatrixTests,
    SpMVMpiTest,
    ::testing::ValuesIn(get_mtx_files("./data"))
);

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    ::testing::InitGoogleTest(&argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
        ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    MPI_Finalize();

    return result;
}