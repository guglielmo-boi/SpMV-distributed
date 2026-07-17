#ifndef SPMV_CSR_ADAPTIVE_CUH
#define SPMV_CSR_ADAPTIVE_CUH

#include "csr_matrix.hpp"
#include "dense_vector.hpp"
#include "spmv_common.cuh"

Metrics spmv_csr_adaptive(const CsrMatrix& A, const DenseVector& x, DenseVector& y);

#endif