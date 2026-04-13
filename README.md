# Neural Network + Mini Autograd (Scalar → Tensor)

This project implements a neural network framework from scratch in C++, starting with a **scalar autograd engine** and extending it to a **vector/tensor-based autograd system with OpenMP parallelization**.

The goal is to understand:
- Computational graphs  
- Automatic differentiation (autograd)  
- Neural network training  
- Scaling from scalar → tensor operations  
- Performance improvements via parallelism  

I have used Breast Cancer Dataset for training and testing my model

---
## Compilation
Here's a bit about my environment. I work on Debian System and my all my .sh files are tested on linux machine. If your OS is different, you may have a different way of running it and you can make those changes.

```
sh compile.sh
```
---
## Run 
This files runs the scalar and vector model on the breast cancer dataset and outputs the loss, training and testing accuracy and training time

```
sh run_scalar_vs_vector_for_benchmark.sh data/breast_cancer_data.csv 0.05 60 8
```
Usage: `sh run_scalar_vs_vector_for_benchmark.sh <data_file> <learning_rate> <epochs> [threads]`

Note: When you run scalar version with high epochs there's a chance you may run out of memory. Sicne I got good results in 60 epochs, I didn't make any changes to accomodate high epoch scenarios. I have 16GB RAM on my machine.

---

## Tests
I made a few sh files to run bad input scenarios. In case of bad input scenarios my code exits gracefully and displays the right error message. You can check those scenarios by running the following tests:
```
sh run_scalar_tests.sh
sh run_tensor_tests.sh
```
---

## Results (Breast Cancer Dataset)

| Version                     | Threads | Train Acc | Test Acc | Time (s) |
|-----------------------------|--------:|----------:|---------:|---------:|
| Scalar Autograd             | 1       | 0.9407    | 0.9386   | 8.07     |
| Tensor Autograd             | 1       | 0.8791    | 0.9035   | 0.024    |
| Tensor Autograd (OpenMP)    | 8       | 0.8791    | 0.9035   | 0.019    |

---

## Repo Organization

You can find all the source code under src/ directory. All the dataset including cleaned breast cancer dataset can be found under data/ directory. All the .sh files are in root and so are compiled binaries.

Repository Link: https://github.com/kavyaas110/neural_network_plus_mini_autograd
