#include <gtest/gtest.h>

#include "csr_matrix.hpp"
#include "dense_vector.hpp"
#include "spmv_cusparse.cuh"
#include "test_utils.hpp"

#include <filesystem>
#include <vector>
#include <string>

// This code was created with the help of generative artificial intelligence.

class SpMVTest : public ::testing::TestWithParam<std::string>
{
protected:
    std::string filename;
    std::unique_ptr<CsrMatrix> A;
    DenseVector x;

    void SetUp() override {
        filename = GetParam();
        A = std::make_unique<CsrMatrix>(MtxParser::parseMtxFile(filename));
        x = DenseVector::random_vector(A->rows);
    }
};

TEST_P(SpMVTest, cuSPARSEMatchesCPU) {
    DenseVector y_cpu = (*A) * x;
    DenseVector y_gpu(A->rows);
    spmv_cusparse(*A, x, y_gpu);

    EXPECT_TRUE(y_cpu.is_close(y_gpu)) << "cuSPARSE mismatch for file: " << filename;
}

INSTANTIATE_TEST_SUITE_P(
    MatrixTests,
    SpMVTest,
    ::testing::ValuesIn(get_mtx_files("./data"))
);