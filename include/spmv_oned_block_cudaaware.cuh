#ifndef SPMV_ONED_BLOCK_CUDAAWARE_CUH
#define SPMV_ONED_BLOCK_CUDAAWARE_CUH

#include "metrics_mpi.hpp"
#include "mtx_parser.hpp"
#include "dense_vector.hpp"

void spmv_oned_block_cudaaware(const MtxParser::MtxMatrix& global_matrix, DenseVector& global_x, DenseVector& global_y, MetricsMpi& metrics_mpi);

#endif