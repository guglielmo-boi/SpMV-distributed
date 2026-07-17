#include "metrics.hpp"

#include <iomanip>

const std::string Metrics::header = "matrix_id,total_execution_time,kernel_execution_time,total_gflops,kernel_gflops";

std::ostream& operator<<(std::ostream& os, const Metrics& metrics) {
    os << metrics.matrix_id << "," << std::fixed << std::setprecision(3)
    << metrics.total_execution_time << ","
    << metrics.kernel_execution_time << ","
    << metrics.total_gflops <<  ","
    << metrics.kernel_gflops;

    return os;
}