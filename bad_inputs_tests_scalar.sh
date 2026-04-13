#!/bin/bash

echo "===== Scalar Tests ====="

echo
echo "Test 1: no args"
./nn_autograd || true

echo
echo "Test 2: incomplete train args"
./nn_autograd train || true

echo
echo "Test 3: incomplete predict args"
./nn_autograd predict || true

echo
echo "Test 4: wrong command"
./nn_autograd hello || true

echo
echo "Test 5: missing training file"
./nn_autograd train data/nope.csv models/model.txt || true

echo
echo "Test 6: missing model file for predict"
./nn_autograd predict models/nope.txt data/test.csv || true

echo
echo "Test 7: empty training file"
./nn_autograd train data/empty.csv models/model.txt || true

echo
echo "Test 8: feature mismatch in prediction"
./nn_autograd predict models/model.txt data/bad_test.csv || true

echo
echo "===== Scalar Tests Finished ====="
