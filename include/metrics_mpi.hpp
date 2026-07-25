#ifndef METRICS_MPI_HPP
#define METRICS_MPI_HPP

#include "metrics.hpp"

#include <iostream>
#include <vector>

struct MetricsMpi
{
    MetricsMpi(int world_size);

    std::string matrix_id;
    float total_execution_time = 0.0;
    float total_gflops = 0.0;
    std::vector<Metrics> metrics;

    static std::string get_header(int world_size);
};

std::ostream& operator<<(std::ostream& os, const MetricsMpi& metrics_mpi);

#endif