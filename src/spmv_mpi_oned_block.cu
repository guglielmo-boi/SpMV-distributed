#include "spmv_mpi_oned_block.cuh"

#include "mpi_common.hpp"
#include "spmv_cusparse.cuh"

std::vector<MtxParser::MtxMatrix> partition_matrix_oned_block(const MtxParser::MtxMatrix& mtx_matrix) {
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<MtxParser::MtxMatrix> partitions(world_size);

    // Initialize cols for each partition
    for (int r = 0; r < world_size; ++r) {
        partitions[r].rows = (mtx_matrix.rows + world_size - r - 1) / world_size;
        partitions[r].cols = mtx_matrix.cols;
        partitions[r].nnz  = 0;
    }

    // Distribute COO entries, updating nnz and rows
    for (const auto& entry : mtx_matrix.entries) {
        int block_size = (mtx_matrix.rows + world_size - 1) / world_size;
        int rank = entry.row / block_size;

        auto local_entry = entry;
        local_entry.row -= rank * block_size;   // Convert global row to local row

        partitions[rank].entries.push_back(local_entry);
        partitions[rank].nnz += 1;
    }

    return partitions;
}

void spmv_mpi_oned_block(const MtxParser::MtxMatrix& global_matrix, DenseVector& global_x, DenseVector& global_y) {
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int global_rows = global_matrix.rows;
    MPI_Bcast(&global_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

    int x_size = (int)global_x.size();

    MPI_Bcast(&x_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        global_x = DenseVector(x_size);
    }

    MPI_Bcast(global_x.data(), x_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    CsrMatrix local_A(local_matrix);
    DenseVector local_y(local_A.rows);

    Metrics metrics = spmv_cusparse(local_A, global_x, local_y);

    if (rank == 0) {
        global_y = DenseVector(global_rows);

        int block_size = (global_rows + world_size - 1) / world_size;

        for (int i = 0; i < local_y.size(); ++i) {
            global_y[i] = local_y[i];
        }

        for (int r = 1; r < world_size; ++r) {
            int local_rows = partitions[r].rows;

            std::vector<dtype> buffer(local_rows);

            MPI_Recv(buffer.data(), local_rows, MPI_FLOAT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < local_rows; ++i) {
                global_y[r * block_size + i] = buffer[i];
            }
        }
    } else {
        MPI_Send(local_y.data(), local_y.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
}