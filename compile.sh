#!/bin/bash
set -e

echo "Compiling scalar version for synthethic dataset..."
g++ -std=c++17 -Wall -Wextra -O2 \
src/main.cpp src/value.cpp src/nn.cpp src/utils.cpp \
-o nn_autograd

echo "Compiling scalar benchmark (breast cancer)..."
g++ -std=c++17 -Wall -Wextra -O2 \
src/scalar_breast_cancer.cpp src/value.cpp src/nn.cpp src/data_utils.cpp \
-o scalar_bc

echo "Compiling tensor benchmark (breast cancer)..."
g++ -std=c++17 -Wall -Wextra -O2 -fopenmp \
src/tensor.cpp src/tensor_nn.cpp src/data_utils.cpp src/tensor_breast_cancer.cpp \
-o tensor_bc

echo "Compilation successful."
