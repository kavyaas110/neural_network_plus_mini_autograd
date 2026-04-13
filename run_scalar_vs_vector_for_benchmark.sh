#!/bin/bash
set -e

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: ./run.sh <data_file> <learning_rate> <epochs> [threads]"
    echo "Example: ./run.sh data/breast_cancer_data.csv 0.05 60 8"
    exit 1
fi

DATA=$1
LR=$2
EPOCHS=$3
THREADS=${4:-8}   # default = 8 if not provided

echo
echo "===== Running Benchmarks ====="

echo
echo "========================================"
echo "Scalar Autograd"
echo "========================================"
./scalar_bc $DATA $LR $EPOCHS

echo
echo "========================================"
echo "Tensor Autograd (1 thread - fair)"
echo "========================================"
./tensor_bc $DATA $LR $EPOCHS 1

echo
echo "========================================"
echo "Tensor Autograd ($THREADS threads - parallel)"
echo "========================================"
./tensor_bc $DATA $LR $EPOCHS $THREADS

echo
echo "===== Benchmark Complete ====="
