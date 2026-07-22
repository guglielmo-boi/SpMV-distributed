#include "csr_matrix.hpp"

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

CsrMatrix::CsrMatrix(const MtxParser::MtxMatrix& matrix) {
    auto entries = matrix.entries;

    std::sort(entries.begin(), entries.end());

    this->rows = matrix.rows;
    this->cols = matrix.cols;
    this->nnz = matrix.nnz;

    this->row_ptr.assign(rows + 1, 0);
    this->col_index.resize(nnz);
    this->values.resize(nnz);
    
    for (const auto& e : entries) {
        this->row_ptr[e.row + 1] += 1;
    }

    for (int r = 1; r <= this->rows; ++r) {
        this->row_ptr[r] += this->row_ptr[r - 1];
    }

    std::vector<int> next = this->row_ptr;

    for (const auto& e : entries) {
        int pos = next[e.row]++;
        this->col_index[pos] = e.col;
        this->values[pos] = e.value;
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