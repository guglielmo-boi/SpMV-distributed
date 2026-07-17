#ifndef SPMV_CUSPARSE_CUH
#define SPMV_CUSPARSE_CUH

#include "csr_matrix.hpp"
#include "dense_vector.hpp"
#include "spmv_common.cuh"

Metrics spmv_cusparse(const CsrMatrix& A, const DenseVector& x, DenseVector& y);

#endif