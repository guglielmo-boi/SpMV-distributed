#ifndef CUDA_EVENT_CHRONO_CUH
#define CUDA_EVENT_CHRONO_CUH

#include <cuda_runtime.h>

class CudaEventChrono
{
public:
    CudaEventChrono();
    ~CudaEventChrono();

    float measure_elapsed_milliseconds();

private:
    cudaEvent_t start;
    cudaEvent_t stop;
};

#endif