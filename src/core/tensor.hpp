#pragma once

#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <string>
#include <vector>
#include "types.hpp"

namespace vanguard {

// ---------------------------------------------------------------------
// Minimal non-owning Tensor View for C++/CUDA tensor passing
// ---------------------------------------------------------------------
template <typename T = float>
struct TensorView {
    T* data = nullptr;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    TensorView() = default;

    TensorView(T* ptr, std::initializer_list<size_t> shape_list)
        : data(ptr), shape(shape_list) {
        compute_strides();
    }

    TensorView(T* ptr, const std::vector<size_t>& shape_vec)
        : data(ptr), shape(shape_vec) {
        compute_strides();
    }

    size_t size() const {
        if (shape.empty()) return 0;
        size_t total = 1;
        for (auto s : shape) total *= s;
        return total;
    }

    size_t bytes() const {
        return size() * sizeof(T);
    }

private:
    void compute_strides() {
        strides.resize(shape.size());
        if (shape.empty()) return;
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }
};

} // namespace vanguard
