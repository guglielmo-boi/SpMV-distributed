#include "metrics.hpp"

#include <iomanip>

std::string Metrics::get_header(int rank) {
    std::string rank_str = std::to_string(rank);
    std::string header = rank_str + "_nnz," + rank_str + "_kernel_execution_time," + rank_str + "_kernel_gflops";    
    return header;
}

std::ostream& operator<<(std::ostream& os, const Metrics& metrics) {
    os << std::fixed << std::setprecision(3)
    << metrics.nnz << ","
    << metrics.kernel_execution_time << ","
    << metrics.kernel_gflops;

    return os;
}