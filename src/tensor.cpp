#include "tensor.h"
#include <stdexcept>
#include <unordered_set>
#include <cmath>
#include <omp.h>

Tensor::Tensor(const std::vector<double>& data, const std::vector<int>& shape)
    : data(data), grad(data.size(), 0.0), shape(shape), op(""), _backward([](){}) {
    if ((int)data.size() != numel()) {
        throw std::runtime_error("Tensor data size does not match shape.");
    }
}

Tensor::Tensor(const std::vector<double>& data,
               const std::vector<int>& shape,
               const std::vector<TensorPtr>& prev,
               const std::string& op)
    : data(data), grad(data.size(), 0.0), shape(shape), prev(prev), op(op), _backward([](){}) {
    if ((int)data.size() != numel()) {
        throw std::runtime_error("Tensor data size does not match shape.");
    }
}

int Tensor::numel() const {
    int total = 1;
    for (int d : shape) total *= d;
    return total;
}

void Tensor::zero_grad() {
    #pragma omp parallel for
    for (int i = 0; i < (int)grad.size(); i++) {
        grad[i] = 0.0;
    }
}

void Tensor::backward() {
    if (data.size() != 1) {
        throw std::runtime_error("backward() currently requires scalar output tensor.");
    }

    std::vector<TensorPtr> topo;
    topo.reserve(1024);
    std::unordered_set<Tensor*> visited;
    visited.reserve(1024);

    std::function<void(TensorPtr)> build_topo = [&](TensorPtr t) {
        if (visited.find(t.get()) != visited.end()) return;
        visited.insert(t.get());
        for (const auto& p : t->prev) {
            build_topo(p);
        }
        topo.push_back(t);
    };

    build_topo(std::shared_ptr<Tensor>(this, [](Tensor*){}));

    std::fill(grad.begin(), grad.end(), 0.0);
    grad[0] = 1.0;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}

TensorPtr make_tensor(const std::vector<double>& data, const std::vector<int>& shape) {
    return std::make_shared<Tensor>(data, shape);
}

TensorPtr make_scalar(double x) {
    return std::make_shared<Tensor>(std::vector<double>{x}, std::vector<int>{1});
}

void check_same_shape(const TensorPtr& a, const TensorPtr& b) {
    if (a->shape != b->shape) {
        throw std::runtime_error("Shape mismatch.");
    }
}

int flat_index_2d(int r, int c, int cols) {
    return r * cols + c;
}

TensorPtr add(TensorPtr a, TensorPtr b) {
    check_same_shape(a, b);

    std::vector<double> out_data(a->data.size());

    #pragma omp parallel for
    for (int i = 0; i < (int)a->data.size(); i++) {
        out_data[i] = a->data[i] + b->data[i];
    }

    auto out = std::make_shared<Tensor>(out_data, a->shape, std::vector<TensorPtr>{a, b}, "add");

    out->_backward = [a, b, out]() {
        for (size_t i = 0; i < out->grad.size(); i++) {
            a->grad[i] += out->grad[i];
            b->grad[i] += out->grad[i];
        }
    };

    return out;
}

TensorPtr sub(TensorPtr a, TensorPtr b) {
    check_same_shape(a, b);

    std::vector<double> out_data(a->data.size());

    #pragma omp parallel for
    for (int i = 0; i < (int)a->data.size(); i++) {
        out_data[i] = a->data[i] - b->data[i];
    }

    auto out = std::make_shared<Tensor>(out_data, a->shape, std::vector<TensorPtr>{a, b}, "sub");

    out->_backward = [a, b, out]() {
        for (size_t i = 0; i < out->grad.size(); i++) {
            a->grad[i] += out->grad[i];
            b->grad[i] -= out->grad[i];
        }
    };

    return out;
}

TensorPtr mul(TensorPtr a, TensorPtr b) {
    check_same_shape(a, b);

    std::vector<double> out_data(a->data.size());

    #pragma omp parallel for
    for (int i = 0; i < (int)a->data.size(); i++) {
        out_data[i] = a->data[i] * b->data[i];
    }

    auto out = std::make_shared<Tensor>(out_data, a->shape, std::vector<TensorPtr>{a, b}, "mul");

    out->_backward = [a, b, out]() {
        for (size_t i = 0; i < out->grad.size(); i++) {
            a->grad[i] += b->data[i] * out->grad[i];
            b->grad[i] += a->data[i] * out->grad[i];
        }
    };

    return out;
}

TensorPtr tanh_act(TensorPtr a) {
    std::vector<double> out_data(a->data.size());

    #pragma omp parallel for
    for (int i = 0; i < (int)a->data.size(); i++) {
        out_data[i] = std::tanh(a->data[i]);
    }

    auto out = std::make_shared<Tensor>(out_data, a->shape, std::vector<TensorPtr>{a}, "tanh");

    out->_backward = [a, out]() {
        for (size_t i = 0; i < out->grad.size(); i++) {
            double t = out->data[i];
            a->grad[i] += (1.0 - t * t) * out->grad[i];
        }
    };

    return out;
}

TensorPtr sum(TensorPtr a) {
    double s = 0.0;

    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < (int)a->data.size(); i++) {
        s += a->data[i];
    }

    auto out = std::make_shared<Tensor>(
        std::vector<double>{s},
        std::vector<int>{1},
        std::vector<TensorPtr>{a},
        "sum"
    );

    out->_backward = [a, out]() {
        for (size_t i = 0; i < a->grad.size(); i++) {
            a->grad[i] += out->grad[0];
        }
    };

    return out;
}

TensorPtr transpose(TensorPtr a) {
    if (a->shape.size() != 2) {
        throw std::runtime_error("transpose() requires a 2D tensor.");
    }

    int rows = a->shape[0];
    int cols = a->shape[1];
    std::vector<double> out_data(a->data.size());

    #pragma omp parallel for collapse(2)
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            out_data[flat_index_2d(c, r, rows)] = a->data[flat_index_2d(r, c, cols)];
        }
    }

    auto out = std::make_shared<Tensor>(
        out_data,
        std::vector<int>{cols, rows},
        std::vector<TensorPtr>{a},
        "transpose"
    );

    out->_backward = [a, out, rows, cols]() {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                a->grad[flat_index_2d(r, c, cols)] += out->grad[flat_index_2d(c, r, rows)];
            }
        }
    };

    return out;
}

TensorPtr matmul(TensorPtr a, TensorPtr b) {
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::runtime_error("matmul() requires 2D tensors.");
    }

    int m = a->shape[0];
    int k1 = a->shape[1];
    int k2 = b->shape[0];
    int n = b->shape[1];

    if (k1 != k2) {
        throw std::runtime_error("matmul() shape mismatch.");
    }

    std::vector<double> out_data(m * n, 0.0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < k1; k++) {
                s += a->data[flat_index_2d(i, k, k1)] * b->data[flat_index_2d(k, j, n)];
            }
            out_data[flat_index_2d(i, j, n)] = s;
        }
    }

    auto out = std::make_shared<Tensor>(
        out_data,
        std::vector<int>{m, n},
        std::vector<TensorPtr>{a, b},
        "matmul"
    );

    out->_backward = [a, b, out, m, n, k1]() {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double g = out->grad[flat_index_2d(i, j, n)];
                for (int k = 0; k < k1; k++) {
                    a->grad[flat_index_2d(i, k, k1)] += b->data[flat_index_2d(k, j, n)] * g;
                    b->grad[flat_index_2d(k, j, n)] += a->data[flat_index_2d(i, k, k1)] * g;
                }
            }
        }
    };

    return out;
}
