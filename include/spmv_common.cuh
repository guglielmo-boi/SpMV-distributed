#ifndef SPMV_COMMON_CUH
#define SPMV_COMMON_CUH

#include "common.hpp"
#include "cuda_event_chrono.cuh"
#include "metrics.hpp"

#include <cuda_runtime.h>

#define WARP_SIZE 32

__device__
dtype warp_reduce_sum(dtype val);

#endif