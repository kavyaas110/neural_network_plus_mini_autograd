#include "value.h"

Value::Value(double data) : data(data), grad(0.0), op(""), _backward([](){}) {}

Value::Value(double data, const std::vector<Value*>& prev, const std::string& op)
    : data(data), grad(0.0), prev(prev), op(op), _backward([](){}) {}

void Value::backward() {
    // to implement
}

Value operator+(Value& a, Value& b) {
    Value out(a.data + b.data, {&a, &b}, "+");
    return out;
}

Value operator*(Value& a, Value& b) {
    Value out(a.data * b.data, {&a, &b}, "*");
    return out;
}

Value tanh(Value& v) {
    Value out(v.data, {&v}, "tanh");
    return out;
}
