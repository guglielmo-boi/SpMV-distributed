#include "spmv_csr_vector.cuh"

__global__
void csr_vector_kernel(
    int rows,
    const int* row_ptr,
    const int* col_idx,
    const dtype* values,
    const dtype* x,
    dtype* y,
    const int* active_rows,
    int num_active)
{
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (warp_id >= num_active) {
        return;
    }

    int row = active_rows[warp_id];
    int row_start = row_ptr[row];
    int row_end   = row_ptr[row + 1];

    dtype sum = 0;

    for (int j = row_start + lane; j < row_end; j += WARP_SIZE) {
        sum += values[j] * x[col_idx[j]];
    }

    unsigned mask = __activemask();

    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    if (lane == 0) {
        y[row] += sum;
    }
}

Metrics spmv_csr_vector(const CsrMatrix& A, const DenseVector& x, DenseVector& y) {
    Metrics metrics;
    CudaEventChrono csr_vector_chrono;

    auto view = A.copy_to_device();
    dtype* d_x = x.copy_to_device();

    dtype* d_y;
    cudaMalloc(&d_y, A.rows * sizeof(dtype));
    cudaMemset(d_y, 0, A.rows * sizeof(dtype));

    std::vector<int> active_rows(A.rows);

    for (int i = 0; i < A.rows; ++i) {
        active_rows[i] = i;
    }

    int* d_active_rows;
    cudaMalloc(&d_active_rows, A.rows * sizeof(int));
    cudaMemcpy(d_active_rows, active_rows.data(), A.rows * sizeof(int), cudaMemcpyHostToDevice);

    int num_active = A.rows;
    int threads = 128;
    int warps_per_block = threads / WARP_SIZE;
    int blocks = (num_active + warps_per_block - 1) / warps_per_block;

    CudaEventChrono csr_vector_kernel_chrono;

    csr_vector_kernel<<<blocks, threads>>>(
        A.rows,
        view.d_row_ptr,
        view.d_col_index,
        view.d_values,
        d_x,
        d_y,
        d_active_rows,
        num_active
    );

    metrics.kernel_execution_time = csr_vector_kernel_chrono.measure_elapsed_milliseconds();

    cudaDeviceSynchronize();

    y.copy_from_device(d_y, A.rows);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_active_rows);

    metrics.total_execution_time = csr_vector_chrono.measure_elapsed_milliseconds();
    metrics.total_gflops = (A.nnz * 2.0) / (metrics.total_execution_time * 1e6);
    metrics.kernel_gflops = (A.nnz * 2.0) / (metrics.kernel_execution_time * 1e6);

    return metrics;
}