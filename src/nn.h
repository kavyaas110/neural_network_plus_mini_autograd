#ifndef NN_H
#define NN_H

#include "value.h"
#include <vector>
#include <memory>

class Neuron {
public:
    std::vector<ValuePtr> w;
    ValuePtr b;
    int nin;

    Neuron(int nin);
    ValuePtr operator()(const std::vector<ValuePtr>& x);
    std::vector<ValuePtr> parameters();
};

class Layer {
public:
    std::vector<Neuron> neurons;
    int nin;
    int nout;

    Layer(int nin, int nout);
    std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& x);
    std::vector<ValuePtr> parameters();
};

class MLP {
public:
    int nin;
    std::vector<int> nouts;
    std::vector<Layer> layers;

    MLP(int nin, const std::vector<int>& nouts);
    std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& x);
    std::vector<ValuePtr> parameters();
};

#endif
