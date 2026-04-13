#include "tensor_nn.h"
#include <cstdlib>

static double rand_double() {
    return 0.1 * ((double) std::rand() / RAND_MAX) * 2.0 - 1.0;
}

Linear::Linear(int in_features, int out_features)
    : in_features(in_features), out_features(out_features) {

    std::vector<double> w_data(in_features * out_features);
    for (auto& v : w_data) v = rand_double();

    std::vector<double> b_data(out_features);
    for (auto& v : b_data) v = rand_double();

    W = make_tensor(w_data, {in_features, out_features});
    b = make_tensor(b_data, {1, out_features});
}

TensorPtr Linear::operator()(TensorPtr x) {
    auto out = matmul(x, W); // (batch, out)

    int batch = out->shape[0];
    int out_dim = out->shape[1];

    std::vector<double> new_data(out->data);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < out_dim; j++) {
            new_data[i * out_dim + j] += b->data[j];
        }
    }

    auto result = make_tensor(new_data, out->shape);

    TensorPtr bias = b;

    result->prev = {out, bias};
    result->_backward = [out, bias, result, batch, out_dim]() {
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < out_dim; j++) {
                double g = result->grad[i * out_dim + j];
                out->grad[i * out_dim + j] += g;
                bias->grad[j] += g;
            }
        }
    };

    return result;
}

std::vector<TensorPtr> Linear::parameters() {
    return {W, b};
}

TensorMLP::TensorMLP(int in_features, const std::vector<int>& sizes) {
    int prev = in_features;
    for (int s : sizes) {
        layers.emplace_back(prev, s);
        prev = s;
    }
}

TensorPtr TensorMLP::operator()(TensorPtr x) {
    TensorPtr out = x;
    for (size_t i = 0; i < layers.size(); i++) {
        out = layers[i](out);
        if (i != layers.size() - 1) {
            out = tanh_act(out);
        }
    }
    return out;
}

std::vector<TensorPtr> TensorMLP::parameters() {
    std::vector<TensorPtr> params;
    for (auto& layer : layers) {
        auto p = layer.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}
