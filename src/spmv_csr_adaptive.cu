#include "spmv_csr_adaptive.cuh"

#include "spmv_csr_stream.cuh"
#include "spmv_csr_vector.cuh"

Metrics spmv_csr_adaptive(const CsrMatrix& A, const DenseVector& x, DenseVector& y) {
    Metrics metrics;
    CudaEventChrono csr_adaptive_chrono;

    std::vector<int> long_rows;
    int threshold = WARP_SIZE;

    for (int i = 0; i < A.rows; ++i) {
        int row_len = A.row_ptr[i + 1] - A.row_ptr[i];

        if (row_len >= threshold) {
            long_rows.push_back(i);
        }
    }

    int num_long = long_rows.size();

    std::vector<char> is_long(A.rows, 0);

    for (int r : long_rows) {
        is_long[r] = 1;
    }

    auto view = A.copy_to_device();
    dtype* d_x = x.copy_to_device();

    dtype* d_y;
    cudaMalloc(&d_y, A.rows * sizeof(dtype));
    cudaMemset(d_y, 0, A.rows * sizeof(dtype));

    int* d_long_rows = nullptr;

    if (num_long > 0) {
        cudaMalloc(&d_long_rows, num_long * sizeof(int));
        cudaMemcpy(d_long_rows, long_rows.data(), num_long * sizeof(int), cudaMemcpyHostToDevice);
    }

    char* d_is_long;
    cudaMalloc(&d_is_long, A.rows * sizeof(char));
    cudaMemcpy(d_is_long, is_long.data(), A.rows * sizeof(char), cudaMemcpyHostToDevice);

    if (num_long > 0) {
        int threads = 128;
        int warps_per_block = threads / WARP_SIZE;
        int blocks = (num_long + warps_per_block - 1) / warps_per_block;

        CudaEventChrono csr_vector_kernel_chrono;

        csr_vector_kernel<<<blocks, threads>>>(
            A.rows,
            view.d_row_ptr,
            view.d_col_index,
            view.d_values,
            d_x,
            d_y,
            d_long_rows,
            num_long
        );

        metrics.kernel_execution_time += csr_vector_kernel_chrono.measure_elapsed_milliseconds();
    }

    int threads = 128;
    int warps_per_block = threads / WARP_SIZE;
    int total_warps = (A.nnz + WARP_SIZE - 1) / WARP_SIZE;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    CudaEventChrono csr_stream_kernel_chrono;

    csr_stream_kernel<<<blocks, threads>>>(
        A.rows,
        A.nnz,
        view.d_row_ptr,
        view.d_col_index,
        view.d_values,
        d_x,
        d_y,
        d_is_long
    );

    metrics.kernel_execution_time += csr_stream_kernel_chrono.measure_elapsed_milliseconds();

    cudaDeviceSynchronize();

    y.copy_from_device(d_y, A.rows);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_is_long);

    if (d_long_rows) {
        cudaFree(d_long_rows);
    }

    metrics.total_execution_time = csr_adaptive_chrono.measure_elapsed_milliseconds();
    metrics.total_gflops = (A.nnz * 2.0) / (metrics.total_execution_time * 1e6);
    metrics.kernel_gflops = (A.nnz * 2.0) / (metrics.kernel_execution_time * 1e6);

    return metrics;
}