#include "mpi_common.hpp"

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