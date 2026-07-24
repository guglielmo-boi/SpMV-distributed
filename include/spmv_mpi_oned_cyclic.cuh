#ifndef SPMV_MPI_ONED_CYCLIC_CUH
#define SPMV_MPI_ONED_CYCLIC_CUH

#include "spmv_cusparse.cuh"

void spmv_mpi_oned_cyclic(const MtxParser::MtxMatrix& global_matrix, DenseVector& global_x, DenseVector& global_y);

#endif