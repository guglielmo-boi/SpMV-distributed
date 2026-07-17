#include "spmv_cusparse.cuh"

#include <cusparse.h>

// This code was created with the help of generative artificial intelligence.

Metrics spmv_cusparse(const CsrMatrix& A, const DenseVector& x, DenseVector& y) {
    Metrics metrics;
    CudaEventChrono cusparse_chrono;

    auto view = A.copy_to_device();
    dtype* d_x = x.copy_to_device();

    dtype* d_y;
    cudaMalloc(&d_y, A.rows * sizeof(dtype));
    cudaMemset(d_y, 0, A.rows * sizeof(dtype));

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;

    cusparseCreateCsr(
        &matA,
        A.rows,
        A.cols,
        A.nnz,
        view.d_row_ptr,
        view.d_col_index,
        view.d_values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    );

    cusparseCreateDnVec(
        &vecX,
        A.cols,
        d_x,
        CUDA_R_32F
    );

    cusparseCreateDnVec(
        &vecY,
        A.rows,
        d_y,
        CUDA_R_32F
    );

    dtype alpha = 1.0;
    dtype beta = 0.0;
    size_t buffer_size = 0;

    cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &buffer_size
    );

    void* d_buffer = nullptr;
    cudaMalloc(&d_buffer, buffer_size);

    CudaEventChrono cusparse_kernel_chrono;

    cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        d_buffer
    );

    metrics.kernel_execution_time = cusparse_kernel_chrono.measure_elapsed_milliseconds();

    cudaDeviceSynchronize();

    y.copy_from_device(d_y, A.rows);

    cudaFree(d_buffer);
    cudaFree(d_x);
    cudaFree(d_y);

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    metrics.total_execution_time = cusparse_chrono.measure_elapsed_milliseconds();
    metrics.total_gflops = (A.nnz * 2.0) / (metrics.total_execution_time * 1e6);
    metrics.kernel_gflops = (A.nnz * 2.0) / (metrics.kernel_execution_time * 1e6);

    return metrics;
}