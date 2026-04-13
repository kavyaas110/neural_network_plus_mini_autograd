#ifndef TENSOR_NN_H
#define TENSOR_NN_H

#include "tensor.h"
#include <vector>

class Linear {
public:
    int in_features;
    int out_features;

    TensorPtr W; // shape: (in, out)
    TensorPtr b; // shape: (1, out)

    Linear(int in_features, int out_features);

    TensorPtr operator()(TensorPtr x); // x: (batch, in)
    std::vector<TensorPtr> parameters();
};

class TensorMLP {
public:
    std::vector<Linear> layers;

    TensorMLP(int in_features, const std::vector<int>& sizes);

    TensorPtr operator()(TensorPtr x);
    std::vector<TensorPtr> parameters();
};

#endif
