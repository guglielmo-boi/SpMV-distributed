#ifndef METRICS_HPP
#define METRICS_HPP

#include <iostream>

struct Metrics
{
    int nnz = 0;
    float kernel_execution_time = 0.0;
    float kernel_gflops = 0.0;
    
    static std::string get_header(int rank);
};

std::ostream& operator<<(std::ostream& os, const Metrics& metrics);

#endif