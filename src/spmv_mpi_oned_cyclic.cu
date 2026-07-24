#include "spmv_mpi_oned_cyclic.cuh"

#include <mpi.h>

#include <vector>

std::vector<MtxParser::MtxMatrix> partition_matrix_oned_cyclic(const MtxParser::MtxMatrix& mtx_matrix) {
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
        int rank = entry.row % world_size;

        auto local_entry = entry;
        local_entry.row /= world_size;   // Convert global row to local row

        partitions[rank].entries.push_back(local_entry);
        partitions[rank].nnz += 1;
    }

    return partitions;
}

void send_local_matrix(const MtxParser::MtxMatrix& local_matrix, int destination) {
    MPI_Send(&local_matrix.rows, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
    MPI_Send(&local_matrix.cols, 1, MPI_INT, destination, 1, MPI_COMM_WORLD);
    MPI_Send(&local_matrix.nnz, 1, MPI_INT, destination, 2, MPI_COMM_WORLD);
    MPI_Send(local_matrix.entries.data(), local_matrix.nnz * sizeof(MtxParser::COOEntry), MPI_BYTE, destination, 3, MPI_COMM_WORLD);
}

MtxParser::MtxMatrix receive_local_matrix(int source) {
    MtxParser::MtxMatrix local_matrix;

    MPI_Recv(&local_matrix.rows, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&local_matrix.cols, 1, MPI_INT, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&local_matrix.nnz, 1, MPI_INT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    local_matrix.entries.resize(local_matrix.nnz);

    MPI_Recv(local_matrix.entries.data(), local_matrix.nnz * sizeof(MtxParser::COOEntry), MPI_BYTE, source, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    return local_matrix;
}

void spmv_mpi_oned_cyclic(const MtxParser::MtxMatrix& global_matrix, DenseVector& global_x, DenseVector& global_y) {
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int global_rows = global_matrix.rows;
    MPI_Bcast(&global_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MtxParser::MtxMatrix local_matrix;
    std::vector<MtxParser::MtxMatrix> partitions;

    if (rank == 0) {
        partitions = partition_matrix_oned_cyclic(global_matrix);
        local_matrix = partitions[0];

        for (int r = 1; r < world_size; ++r) {
            send_local_matrix(partitions[r], r);
        }
    } else {
        local_matrix = receive_local_matrix(0);
    }

    MPI_Bcast(global_x.data(), global_x.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    CsrMatrix local_A(local_matrix);
    DenseVector local_y(local_A.rows);

    Metrics metrics = spmv_cusparse(local_A, global_x, local_y);

    if (rank == 0) {
        global_y = DenseVector(global_rows);

        for (int i = 0; i < local_y.size(); ++i) {
            global_y[i * world_size] = local_y[i];
        }

        for (int r = 1; r < world_size; ++r) {
            int local_rows = partitions[r].rows;

            std::vector<dtype> buffer(local_rows);

            MPI_Recv(buffer.data(), local_rows, MPI_FLOAT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < local_rows; ++i) {
                global_y[i * world_size + r] = buffer[i];
            }
        }
    } else {
        MPI_Send(local_y.data(), local_y.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
}