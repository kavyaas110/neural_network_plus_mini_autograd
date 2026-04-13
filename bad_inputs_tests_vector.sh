#!/bin/bash

echo "===== Tensor Bad Input + Sanity Tests ====="

echo
echo "Test 1: no args"
./tensor_bc || true

echo
echo "Test 2: too few args"
./tensor_bc data/breast_cancer_data.csv 0.05 || true

echo
echo "Test 3: missing file"
./tensor_bc data/nope.csv 0.05 200 || true

echo
echo "Test 4: bad learning rate (non-numeric)"
./tensor_bc data/breast_cancer_data.csv abc 200 || true

echo
echo "Test 5: bad epochs (non-numeric)"
./tensor_bc data/breast_cancer_data.csv 0.05 xyz || true

echo
echo "Test 6: negative learning rate"
./tensor_bc data/breast_cancer_data.csv -1 200 || true

echo
echo "Test 7: zero learning rate"
./tensor_bc data/breast_cancer_data.csv 0 200 || true

echo
echo "Test 8: zero epochs"
./tensor_bc data/breast_cancer_data.csv 0.05 0 || true

echo
echo "Test 9: negative epochs"
./tensor_bc data/breast_cancer_data.csv 0.05 -10 || true

echo
echo "Test 10: bad thread count (zero)"
./tensor_bc data/breast_cancer_data.csv 0.05 200 0 || true

echo
echo "Test 11: bad thread count (negative)"
./tensor_bc data/breast_cancer_data.csv 0.05 200 -2 || true

echo
echo "Test 12: bad thread count (non-numeric)"
./tensor_bc data/breast_cancer_data.csv 0.05 200 abc || true

echo
echo "===== Tensor Tests Finished ====="
