#include "nn.h"
#include <cstdlib>

static double rand_double() {
    return ((double) std::rand() / RAND_MAX) * 2.0 - 1.0;
}

Neuron::Neuron(int nin) : nin(nin) {
    for (int i = 0; i < nin; i++) {
        w.push_back(make_value(rand_double()));
    }
    b = make_value(rand_double());
}

ValuePtr Neuron::operator()(const std::vector<ValuePtr>& x) {
    ValuePtr act = b;
    for (int i = 0; i < nin; i++) {
        act = add(act, mul(w[i], x[i]));
    }
    return tanh_act(act);
}

std::vector<ValuePtr> Neuron::parameters() {
    std::vector<ValuePtr> params = w;
    params.push_back(b);
    return params;
}

Layer::Layer(int nin, int nout) : nin(nin), nout(nout) {
    for (int i = 0; i < nout; i++) {
        neurons.emplace_back(nin);
    }
}

std::vector<ValuePtr> Layer::operator()(const std::vector<ValuePtr>& x) {
    std::vector<ValuePtr> out;
    for (auto& neuron : neurons) {
        out.push_back(neuron(x));
    }
    return out;
}

std::vector<ValuePtr> Layer::parameters() {
    std::vector<ValuePtr> params;
    for (auto& neuron : neurons) {
        auto p = neuron.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

MLP::MLP(int nin, const std::vector<int>& nouts) : nin(nin), nouts(nouts) {
    std::vector<int> sz;
    sz.push_back(nin);
    for (int x : nouts) sz.push_back(x);

    for (size_t i = 0; i < nouts.size(); i++) {
        layers.emplace_back(sz[i], sz[i + 1]);
    }
}

std::vector<ValuePtr> MLP::operator()(const std::vector<ValuePtr>& x) {
    std::vector<ValuePtr> out = x;
    for (auto& layer : layers) {
        out = layer(out);
    }
    return out;
}

std::vector<ValuePtr> MLP::parameters() {
    std::vector<ValuePtr> params;
    for (auto& layer : layers) {
        auto p = layer.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}
