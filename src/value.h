#ifndef VALUE_H
#define VALUE_H

#include <vector>
#include <string>
#include <functional>
#include <memory>

class Value {
public:
    double data;
    double grad;
    std::vector<std::shared_ptr<Value>> prev;
    std::string op;
    std::function<void()> _backward;

    Value(double data);
    Value(double data, const std::vector<std::shared_ptr<Value>>& prev, const std::string& op);

    void backward();
};

using ValuePtr = std::shared_ptr<Value>;

ValuePtr make_value(double x);

ValuePtr add(ValuePtr a, ValuePtr b);
ValuePtr mul(ValuePtr a, ValuePtr b);
ValuePtr tanh_act(ValuePtr v);
ValuePtr sub(ValuePtr a, ValuePtr b);
ValuePtr neg(ValuePtr v);
ValuePtr square(ValuePtr v);

#endif
