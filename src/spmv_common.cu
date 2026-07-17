#include "spmv_common.cuh"

// Warp-level reduction
__device__
dtype warp_reduce_sum(dtype val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(__activemask(), val, offset);
    }
        
    return val;
}