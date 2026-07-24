#ifndef MPI_COMMON_HPP
#define MPI_COMMON_HPP

#include "mtx_parser.hpp"

#include <mpi.h>

void send_local_matrix(const MtxParser::MtxMatrix& local_matrix, int destination);

MtxParser::MtxMatrix receive_local_matrix(int source);

#endif