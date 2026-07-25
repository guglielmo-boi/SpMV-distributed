#include "metrics_mpi.hpp"

#include <iomanip>

MetricsMpi::MetricsMpi(int world_size) :
    metrics(world_size) {}

std::string MetricsMpi::get_header(int world_size) {
    std::string header = "matrix_id,total_execution_time,total_gflops";

    for (int r = 0; r < world_size; ++r) {
        header += "," + Metrics::get_header(r);
    }

    return header;
}

std::ostream& operator<<(std::ostream& os, const MetricsMpi& metrics_mpi) {
    os << metrics_mpi.matrix_id << "," << std::fixed << std::setprecision(3)
    << metrics_mpi.total_execution_time << ","
    << metrics_mpi.total_gflops;

    for (const auto& metrics : metrics_mpi.metrics) {
        os << "," << metrics;
    }

    return os;
}