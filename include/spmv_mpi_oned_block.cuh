#ifndef SPMV_MPI_ONED_BLOCK_CUH
#define SPMV_MPI_ONED_BLOCK_CUH

#include "mtx_parser.hpp"
#include "dense_vector.hpp"
#include "spmv_common.cuh"

void spmv_mpi_oned_block(const MtxParser::MtxMatrix& global_matrix, DenseVector& global_x, DenseVector& global_y);

#endif