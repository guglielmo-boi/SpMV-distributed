#include "spmv_oned_block_cudaaware.cuh"

#include "csr_matrix.hpp"
#include "spmv_common.cuh"
#include "mpi_common.hpp"
#include "spmv_oned_block.cuh"

#include <cusparse.h>
#include <numeric>

// This code was created with the help of generative artificial intelligence.

void spmv_oned_block_cudaaware(const MtxParser::MtxMatrix& global_matrix, DenseVector& global_x, DenseVector& global_y, MetricsMpi& metrics_mpi) {
    CudaEventChrono total_chrono;

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int global_rows = (rank == 0) ? global_matrix.rows : 0;
    int global_cols = (rank == 0) ? global_matrix.cols : 0;
    MPI_Bcast(&global_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MtxParser::MtxMatrix local_matrix;
    std::vector<MtxParser::MtxMatrix> partitions;

    if (rank == 0) {
        partitions = partition_matrix_oned_block(global_matrix);
        local_matrix = partitions[0];

        for (int r = 1; r < world_size; ++r) {
            send_local_matrix(partitions[r], r);
        }
    } else {
        local_matrix = receive_local_matrix(0);
    }

    dtype* d_x = nullptr;
    cudaMalloc(&d_x, global_cols * sizeof(dtype));

    if (rank == 0) {
        cudaMemcpy(d_x, global_x.data(), global_cols * sizeof(dtype), cudaMemcpyHostToDevice);
    }

    MPI_Bcast(d_x, global_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    CsrMatrix local_A(local_matrix);
    auto view = local_A.copy_to_device();

    dtype* d_y = nullptr;
    cudaMalloc(&d_y, local_A.rows * sizeof(dtype));
    cudaMemset(d_y, 0, local_A.rows * sizeof(dtype));

    Metrics metrics;
    {
        CudaEventChrono kernel_chrono;

        cusparseHandle_t handle;
        cusparseCreate(&handle);

        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;

        cusparseCreateCsr(&matA,
            local_A.rows, local_A.cols, local_A.nnz,
            view.d_row_ptr, view.d_col_index, view.d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseCreateDnVec(&vecX, global_cols, d_x, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, local_A.rows, d_y, CUDA_R_32F);

        dtype alpha = 1.0f, beta = 0.0f;
        size_t buf_size = 0;
        cusparseSpMV_bufferSize(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size);

        void* d_buf = nullptr;
        
        if (buf_size > 0) {
            cudaMalloc(&d_buf, buf_size);
        }

        cusparseSpMV(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf);

        cudaDeviceSynchronize();

        metrics.nnz = local_A.nnz;
        metrics.kernel_execution_time = kernel_chrono.measure_elapsed_milliseconds();
        metrics.kernel_gflops = (local_A.nnz * 2.0) / (metrics.kernel_execution_time * 1e6);

        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseDestroy(handle);

        if (d_buf) {
            cudaFree(d_buf);
    
        }
    }

    if (rank == 0) {
        metrics_mpi.metrics[0] = metrics;

        for (int r = 1; r < world_size; ++r) {
            MPI_Recv(&metrics_mpi.metrics[r], sizeof(Metrics), MPI_BYTE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
            
    } else {
        MPI_Send(&metrics, sizeof(Metrics), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }

    int local_rows = local_A.rows;
    std::vector<int> recv_counts(world_size), displs(world_size);
    MPI_Gather(&local_rows, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;

        for (int r = 1; r < world_size; ++r) {
            displs[r] = displs[r-1] + recv_counts[r-1];
        }

        dtype* d_global_y = nullptr;
        cudaMalloc(&d_global_y, global_rows * sizeof(dtype));

        MPI_Gatherv(d_y, local_rows, MPI_FLOAT, d_global_y, recv_counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        global_y = DenseVector(global_rows);
        cudaMemcpy(global_y.data(), d_global_y, global_rows * sizeof(dtype), cudaMemcpyDeviceToHost);
        cudaFree(d_global_y);
    } else {
        MPI_Gatherv(d_y, local_rows, MPI_FLOAT, nullptr, nullptr, nullptr, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    cudaFree(d_x);
    cudaFree(d_y);

    if (rank == 0) {
        metrics_mpi.total_execution_time = total_chrono.measure_elapsed_milliseconds();
        metrics_mpi.total_gflops = (global_matrix.nnz * 2.0) / (metrics_mpi.total_execution_time * 1e6);
    }
}