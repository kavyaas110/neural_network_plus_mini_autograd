#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include "nn.h"
#include "value.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    try {
        std::srand(42);

        if (argc < 2) {
            std::cerr << "Usage:\n";
            std::cerr << "  ./mltool train <train_csv> <model_out>\n";
            std::cerr << "  ./mltool predict <model_file> <test_csv>\n";
            return 1;
        }

        std::string command = argv[1];

        if (command == "train") {
            if (argc < 4) {
                std::cerr << "Usage: ./mltool train <train_csv> <model_out>\n";
                return 1;
            }

            std::string train_file = argv[2];
            std::string model_out = argv[3];

            auto train_data = read_training_csv(train_file);
            auto X = train_data.first;
            auto Y = train_data.second;

            if (X.empty()) {
                throw std::runtime_error("Training data is empty.");
            }

            int nin = X[0].size();
            for (size_t i = 0; i < X.size(); i++) {
                if ((int)X[i].size() != nin) {
                    throw std::runtime_error("Inconsistent number of features in training data at row " + std::to_string(i));
                }
            }

            MLP model(nin, {4, 1});

            double lr = 0.05;
            int epochs = 100;

            for (int epoch = 0; epoch < epochs; epoch++) {
                ValuePtr total_loss = make_value(0.0);

                for (size_t i = 0; i < X.size(); i++) {
                    std::vector<ValuePtr> xvals;
                    for (double x : X[i]) {
                        xvals.push_back(make_value(x));
                    }

                    auto pred = model(xvals)[0];
                    auto target = make_value(Y[i]);

                    auto diff = sub(pred, target);
                    auto loss = square(diff);
                    total_loss = add(total_loss, loss);
                }

                auto params = model.parameters();
                for (auto& p : params) {
                    p->grad = 0.0;
                }

                total_loss->backward();

                for (auto& p : params) {
                    p->data -= lr * p->grad;
                }

                if (epoch % 10 == 0) {
                    std::cout << "Epoch " << epoch
                              << ", Loss = " << total_loss->data << "\n";
                }
            }

            save_model(model, model_out);
            std::cout << "Model saved to " << model_out << "\n";
        }
        else if (command == "predict") {
            if (argc < 4) {
                std::cerr << "Usage: ./mltool predict <model_file> <test_csv>\n";
                return 1;
            }

            std::string model_file = argv[2];
            std::string test_file = argv[3];

            MLP model = load_model(model_file);
            auto X = read_prediction_csv(test_file);

            for (size_t i = 0; i < X.size(); i++) {
                if ((int)X[i].size() != model.nin) {
                    throw std::runtime_error(
                        "Feature size mismatch on row " + std::to_string(i) +
                        ". Expected " + std::to_string(model.nin) +
                        ", got " + std::to_string(X[i].size())
                    );
                }

                std::vector<ValuePtr> xvals;
                for (double x : X[i]) {
                    xvals.push_back(make_value(x));
                }

                auto pred = model(xvals)[0];
                std::cout << pred->data << "\n";
            }
        }
        else {
            std::cerr << "Unknown command: " << command << "\n";
            return 1;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
