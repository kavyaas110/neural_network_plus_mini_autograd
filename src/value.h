#ifndef VALUE_H
#define VALUE_H

#include <vector>
#include <string>
#include <functional>

class Value {
public:
    double data;
    double grad;
    std::vector<Value*> prev;
    std::string op;
    std::function<void()> _backward;

    Value(double data);
    Value(double data, const std::vector<Value*>& prev, const std::string& op);

    void backward();
};

Value operator+(Value& a, Value& b);
Value operator*(Value& a, Value& b);
Value tanh(Value& v);

#endif
