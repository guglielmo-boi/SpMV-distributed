#include "csr_matrix.hpp"

#include "mtx_parser.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

CsrMatrix::DeviceView::DeviceView(const CsrMatrix& matrix) :
    rows(rows), cols(cols), nnz(nnz) {}

CsrMatrix::DeviceView::~DeviceView() {
    cudaFree(this->d_row_ptr);
    cudaFree(this->d_col_index);
    cudaFree(this->d_values);
}

CsrMatrix::CsrMatrix(const std::string& file_path) {
    auto mtx_matrix = MtxParser::parseMtxFile(file_path);

    this->rows = mtx_matrix.rows;
    this->cols = mtx_matrix.cols;
    this->nnz = mtx_matrix.nnz;

    this->row_ptr.resize(mtx_matrix.rows + 1, 0);
    this->col_index.resize(mtx_matrix.nnz, 0);
    this->values.resize(mtx_matrix.nnz, 0.0);
    
    int counter = 0;

    for (const auto& [row, col_value] : mtx_matrix.elements) {
        for (const auto& [col, value] : col_value) {
            this->row_ptr[row] += 1;
            this->col_index[counter] = col - 1;
            this->values[counter++] = value;
        }
    }

    for (int r = 1; r <= this->rows; ++r) {
        this->row_ptr[r] += this->row_ptr[r - 1];
    }
}

const int* CsrMatrix::row_ptr_data() const { 
    return row_ptr.data();
}

const int* CsrMatrix::col_index_data() const {
    return col_index.data();
}

const dtype* CsrMatrix::values_data() const { 
    return values.data(); 
}

CsrMatrix::DeviceView CsrMatrix::copy_to_device() const {
    DeviceView view(*this);

    cudaMalloc(&view.d_row_ptr, (this->rows + 1) * sizeof(int));
    cudaMalloc(&view.d_col_index, this->nnz * sizeof(int));
    cudaMalloc(&view.d_values, this->nnz * sizeof(dtype));

    cudaMemcpy(view.d_row_ptr, this->row_ptr.data(), (this->rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(view.d_col_index, this->col_index.data(), this->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(view.d_values, this->values.data(), this->nnz * sizeof(dtype), cudaMemcpyHostToDevice);

    return view;
}

DenseVector operator*(const CsrMatrix& csr_matrix, const DenseVector& dense_vector) {
    DenseVector ret = DenseVector(dense_vector.size(), 0.0);

    for (int r = 0; r < csr_matrix.rows; ++r) {
        for (int i = csr_matrix.row_ptr[r]; i < csr_matrix.row_ptr[r + 1]; ++i) {
            ret[r] += csr_matrix.values[i] * dense_vector[csr_matrix.col_index[i]]; 
        } 
    }

    return ret;
}