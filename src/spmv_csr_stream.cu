#include "spmv_csr_stream.cuh"

// binary search: find row for NNZ index
__device__
int find_row(const int* row_ptr, int rows, int idx) {
    int left = 0;
    int right = rows;

    while (left < right) {
        int mid = (left + right) / 2;

        if (row_ptr[mid] <= idx) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left - 1;
}

__global__
void csr_stream_kernel(
    int rows,
    int nnz,
    const int* row_ptr,
    const int* col_idx,
    const dtype* values,
    const dtype* x,
    dtype* y,
    const char* is_long)
{
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    int chunk_size = WARP_SIZE;
    int start = warp_id * chunk_size;

    if (start >= nnz) {
        return;
    }

    int end = min(start + chunk_size, nnz);
    dtype val = 0;
    int row = -1;

    if (start + lane < end) {
        int i = start + lane;

        val = values[i] * x[col_idx[i]];
        row = find_row(row_ptr, rows, i);

        if (is_long != nullptr && is_long[row]) {
            val = 0;
            row = -1;
        }
    }

    // segmented reduction inside warp
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        int other_row = __shfl_up_sync(__activemask(), row, offset);
        dtype other_val = __shfl_up_sync(__activemask(), val, offset);

        if (lane >= offset && row == other_row) {
            val += other_val;
        }
    }

    // detect segment end
    int next_row = __shfl_down_sync(__activemask(), row, 1);

    if (lane == WARP_SIZE - 1 || row != next_row) {
        if (row >= 0) {
            atomicAdd(&y[row], val);
        }
    }
}

Metrics spmv_csr_stream(const CsrMatrix& A, const DenseVector& x, DenseVector& y) {
    Metrics metrics;
    CudaEventChrono csr_stream_chrono;

    auto view = A.copy_to_device();
    dtype* d_x = x.copy_to_device();

    dtype* d_y;
    cudaMalloc(&d_y, A.rows * sizeof(dtype));
    cudaMemset(d_y, 0, A.rows * sizeof(dtype));

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
        nullptr
    );

    metrics.kernel_execution_time = csr_stream_kernel_chrono.measure_elapsed_milliseconds();

    cudaDeviceSynchronize();

    y.copy_from_device(d_y, A.rows);

    cudaFree(d_x);
    cudaFree(d_y);

    metrics.total_execution_time = csr_stream_chrono.measure_elapsed_milliseconds();
    metrics.total_gflops = (A.nnz * 2.0) / (metrics.total_execution_time * 1e6);
    metrics.kernel_gflops = (A.nnz * 2.0) / (metrics.kernel_execution_time * 1e6);

    return metrics;
}