#include "value.h"
#include <cmath>
#include <set>
#include <functional>

Value::Value(double data) : data(data), grad(0.0), op(""), _backward([](){}) {}

Value::Value(double data, const std::vector<std::shared_ptr<Value>>& prev, const std::string& op)
    : data(data), grad(0.0), prev(prev), op(op), _backward([](){}) {}

ValuePtr make_value(double x) {
    return std::make_shared<Value>(x);
}

ValuePtr add(ValuePtr a, ValuePtr b) {
    auto out = std::make_shared<Value>(a->data + b->data, std::vector<ValuePtr>{a, b}, "+");

    out->_backward = [a, b, out]() {
        a->grad += 1.0 * out->grad;
        b->grad += 1.0 * out->grad;
    };

    return out;
}

ValuePtr mul(ValuePtr a, ValuePtr b) {
    auto out = std::make_shared<Value>(a->data * b->data, std::vector<ValuePtr>{a, b}, "*");

    out->_backward = [a, b, out]() {
        a->grad += b->data * out->grad;
        b->grad += a->data * out->grad;
    };

    return out;
}

ValuePtr tanh_act(ValuePtr v) {
    double t = std::tanh(v->data);
    auto out = std::make_shared<Value>(t, std::vector<ValuePtr>{v}, "tanh");

    out->_backward = [v, out, t]() {
        v->grad += (1 - t * t) * out->grad;
    };

    return out;
}

ValuePtr neg(ValuePtr v) {
    auto minus_one = make_value(-1.0);
    return mul(v, minus_one);
}

ValuePtr sub(ValuePtr a, ValuePtr b) {
    return add(a, neg(b));
}

ValuePtr square(ValuePtr v) {
    return mul(v, v);
}

void Value::backward() {
    std::vector<ValuePtr> topo;
    std::set<Value*> visited;

    std::function<void(ValuePtr)> build_topo = [&](ValuePtr v) {
        if (visited.find(v.get()) == visited.end()) {
            visited.insert(v.get());
            for (auto child : v->prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(std::shared_ptr<Value>(this, [](Value*){}));

    this->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}
