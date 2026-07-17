#include "dense_vector.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <random>

DenseVector::DenseVector(int size, dtype value) : values(size, value) {}

DenseVector::DenseVector(std::initializer_list<dtype> list) : values(list) {}

DenseVector DenseVector::random_vector(int size) {
    DenseVector ret(size);

    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 1000);

    for (int i = 0; i < size; ++i) {
        ret[i] = static_cast<dtype>(dist(gen));
    }

    return ret;
}

bool DenseVector::is_close(const DenseVector& other, dtype epsilon) const {
    if (this->size() != other.size()) {
        return false;
    }

    for (int i = 0; i < this->size(); ++i) {
        float a = static_cast<float>(this->operator[](i));
        float b = static_cast<float>(other[i]);

        float abs_error = std::fabs(a - b);
        float denom = std::fmaxf(std::fabs(a), std::fabs(b));
        float rel_error = (denom > 0.0) ? (abs_error / denom) : abs_error;

        if (abs_error > epsilon && rel_error > epsilon) {
            return false;
        }
    }

    return true;
}

std::size_t DenseVector::size() const {
    return this->values.size();
}

dtype* DenseVector::data() {
    return this->values.data();
}

const dtype* DenseVector::data() const {
    return this->values.data();
}

dtype* DenseVector::copy_to_device() const {
    dtype* d_values;
    cudaMalloc(&d_values, this->size() * sizeof(dtype));
    cudaMemcpy(d_values, this->values.data(), this->size() * sizeof(dtype), cudaMemcpyHostToDevice);
    return d_values;
}

void DenseVector::copy_from_device(dtype* d_values, std::size_t size) {
    this->values.resize(size);
    cudaMemcpy(this->values.data(), d_values, size * sizeof(dtype), cudaMemcpyDeviceToHost);
}

dtype& DenseVector::operator[](std::size_t idx) {
    return this->values[idx];
}

const dtype& DenseVector::operator[](std::size_t idx) const {
    return this->values[idx];
}

DenseVector& DenseVector::operator+=(const DenseVector& rhs) {
    for (int i = 0; i < this->size(); ++i) {
        (*this)[i] += rhs[i];
    }

    return *this;
}

DenseVector operator+(const DenseVector& lhs, const DenseVector& rhs) {
    DenseVector ret = lhs;
    ret += rhs;
    return ret;
}

DenseVector operator*(const DenseVector& dense_vector, dtype scalar) {
    DenseVector ret = dense_vector;
    
    for (int i = 0; i < ret.size(); ++i) {
        ret[i] *= scalar;
    }

    return ret;
}

DenseVector operator*(dtype scalar, const DenseVector& dense_vector) {
    return dense_vector * scalar;
}