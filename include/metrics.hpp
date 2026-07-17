#ifndef METRICS_HPP
#define METRICS_HPP

#include <iostream>

struct Metrics
{
    std::string matrix_id;
    float total_execution_time = 0.0;
    float kernel_execution_time = 0.0;
    float total_gflops = 0.0;
    float kernel_gflops = 0.0;

    static const std::string header;
};

std::ostream& operator<<(std::ostream& os, const Metrics& metrics);

#endif