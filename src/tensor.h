#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <functional>
#include <memory>

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor {
public:
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<int> shape;
    std::vector<TensorPtr> prev;
    std::string op;
    std::function<void()> _backward;

    Tensor(const std::vector<double>& data, const std::vector<int>& shape);
    Tensor(const std::vector<double>& data,
           const std::vector<int>& shape,
           const std::vector<TensorPtr>& prev,
           const std::string& op);

    int numel() const;
    void zero_grad();
    void backward();
};

TensorPtr make_tensor(const std::vector<double>& data, const std::vector<int>& shape);
TensorPtr make_scalar(double x);

TensorPtr add(TensorPtr a, TensorPtr b);
TensorPtr sub(TensorPtr a, TensorPtr b);
TensorPtr mul(TensorPtr a, TensorPtr b);      // elementwise
TensorPtr tanh_act(TensorPtr a);
TensorPtr sum(TensorPtr a);                   // returns scalar
TensorPtr matmul(TensorPtr a, TensorPtr b);   // 2D x 2D
TensorPtr transpose(TensorPtr a);             // 2D only

void check_same_shape(const TensorPtr& a, const TensorPtr& b);
int flat_index_2d(int r, int c, int cols);

#endif
