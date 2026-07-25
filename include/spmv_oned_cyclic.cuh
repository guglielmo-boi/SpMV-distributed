#ifndef SPMV_ONED_CYCLIC_CUH
#define SPMV_ONED_CYCLIC_CUH

#include "metrics_mpi.hpp"
#include "mtx_parser.hpp"
#include "dense_vector.hpp"

void spmv_oned_cyclic(const MtxParser::MtxMatrix& global_matrix, DenseVector& global_x, DenseVector& global_y, MetricsMpi& metrics_mpi);

#endif