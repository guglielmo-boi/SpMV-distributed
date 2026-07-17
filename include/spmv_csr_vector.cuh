#ifndef SPMV_CSR_VECTOR_CUH
#define SPMV_CSR_VECTOR_CUH

#include "csr_matrix.hpp"
#include "dense_vector.hpp"
#include "spmv_common.cuh"

__global__ 
void csr_vector_kernel(
    int rows,
    const int* row_ptr,
    const int* col_idx,
    const dtype* values,
    const dtype* x,
    dtype* y,
    const int* active_rows, // contains the indices of rows assigned to CSR-Vector
    int num_active // active_rows size
); 

Metrics spmv_csr_vector(const CsrMatrix& A, const DenseVector& x, DenseVector& y);

#endif