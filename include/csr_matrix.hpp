#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include "common.hpp"
#include "dense_vector.hpp"

#include <vector>
#include <string>

class CsrMatrix
{
public:
    struct DeviceView 
    {
        explicit DeviceView(const CsrMatrix& matrix);
        ~DeviceView();

        int rows;
        int cols;
        int nnz;

        int* d_row_ptr;
        int* d_col_index;
        dtype* d_values;
    };

    CsrMatrix() = delete;
    explicit CsrMatrix(const std::string& file_path);

    const int* row_ptr_data() const;
    const int* col_index_data() const;
    const dtype* values_data() const;
    DeviceView copy_to_device() const;

    int rows;
    int cols;
    int nnz;

    std::vector<int> row_ptr;
    std::vector<int> col_index;
    std::vector<dtype> values;
};

DenseVector operator*(const CsrMatrix& csr_matrix, const DenseVector& dense_vector);

#endif