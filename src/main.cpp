#include "csr_matrix.hpp"
#include "dense_vector.hpp"
#include "mtx_parser.hpp"
#include "metrics.hpp"
#include "spmv_mpi_oned_cyclic.cuh"

#include <mpi.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <fstream>

std::vector<std::filesystem::path> get_matrix_files(const std::filesystem::path& data_dir) {
    std::vector<std::filesystem::path> files;

    if (!std::filesystem::exists(data_dir) || !std::filesystem::is_directory(data_dir)) {
        std::cerr << "Input directory does not exist!" << std::endl;
        return {};
    }

    for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mtx") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());

    return files;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(rank % device_count);

    if (rank == 0) {
        if (argc != 3) {
            std::cerr << "Usage: ./spmv <input_directory_path> <output_log_path>" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    std::filesystem::path data_dir(argv[1]);
    auto run_id = std::to_string(std::time(nullptr));

    std::filesystem::path log_dir = std::filesystem::path(argv[2]) / run_id;
    std::filesystem::create_directories(log_dir);

    std::ofstream oned_partition_log(log_dir / "oned_partition.csv");

    oned_partition_log << Metrics::header << '\n';

    auto matrix_files = get_matrix_files(argv[1]);

    for (const auto& matrix_path : matrix_files) {
        MtxParser::MtxMatrix mtx_matrix;
        DenseVector x(mtx_matrix.cols);
        DenseVector y(mtx_matrix.rows);

        if (rank == 0) {
            mtx_matrix = MtxParser::parseMtxFile(matrix_path);
            x = DenseVector::random_vector(mtx_matrix.cols);
        }

        // 1D cyclic partition
        {
            spmv_mpi_oned_cyclic(mtx_matrix, x, y);

            if (rank == 0) {
                CsrMatrix A(mtx_matrix);
                DenseVector y_cpu = A * x;
                std::cout << y_cpu.is_close(y) << std::endl;
            }
        }
    }

    MPI_Finalize();

    return 0;
}