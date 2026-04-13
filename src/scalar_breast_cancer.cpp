#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <stdexcept>
#include <chrono>
#include "nn.h"
#include "value.h"
#include "data_utils.h"

struct NormStats {
    std::vector<double> mean;
    std::vector<double> stddev;
};

static NormStats compute_norm_stats(const std::vector<std::vector<double>>& X) {
    int n = X.size();
    int d = X[0].size();

    NormStats stats;
    stats.mean.assign(d, 0.0);
    stats.stddev.assign(d, 0.0);

    for (int j = 0; j < d; j++) {
        for (int i = 0; i < n; i++) stats.mean[j] += X[i][j];
        stats.mean[j] /= n;

        for (int i = 0; i < n; i++) {
            double diff = X[i][j] - stats.mean[j];
            stats.stddev[j] += diff * diff;
        }
        stats.stddev[j] = std::sqrt(stats.stddev[j] / n);
        if (stats.stddev[j] == 0.0) stats.stddev[j] = 1.0;
    }

    return stats;
}

static void apply_norm_stats(std::vector<std::vector<double>>& X, const NormStats& stats) {
    int n = X.size();
    int d = X[0].size();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            X[i][j] = (X[i][j] - stats.mean[j]) / stats.stddev[j];
        }
    }
}

static double compute_accuracy(MLP& model,
                               const std::vector<std::vector<double>>& X,
                               const std::vector<double>& Y) {
    int correct = 0;

    for (int i = 0; i < (int)X.size(); i++) {
        std::vector<ValuePtr> xvals;
        for (double x : X[i]) {
            xvals.push_back(make_value(x));
        }

        auto pred = model(xvals)[0];
        int pred_label = pred->data > 0 ? 1 : -1;
        int true_label = (int)Y[i];

        if (pred_label == true_label) correct++;
    }

    return (double)correct / Y.size();
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 4) {
            std::cerr << "Usage: ./scalar_bc <csv_file> <learning_rate> <epochs>\n";
            return 1;
        }

        std::string filename = argv[1];

        double lr;
        int epochs;

        try {
            lr = std::stod(argv[2]);
        } catch (...) {
            throw std::runtime_error("Invalid learning rate. Must be a number.");
        }

        try {
            epochs = std::stoi(argv[3]);
        } catch (...) {
            throw std::runtime_error("Invalid epochs value. Must be an integer.");
        }

        if (lr <= 0.0) {
            throw std::runtime_error("Learning rate must be > 0.");
        }
        if (epochs <= 0) {
            throw std::runtime_error("Epochs must be > 0.");
        }

        std::srand(42);

        auto data = read_training_csv(filename);
        auto X = data.first;
        auto Y = data.second;

        if (X.empty()) {
            throw std::runtime_error("Dataset is empty.");
        }

        int n = X.size();
        int d = X[0].size();

        for (int i = 0; i < n; i++) {
            if ((int)X[i].size() != d) {
                throw std::runtime_error("Inconsistent feature count at row " + std::to_string(i));
            }
        }

        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);

        std::mt19937 rng(42);
        std::shuffle(idx.begin(), idx.end(), rng);

        int train_n = (int)(0.8 * n);

        std::vector<std::vector<double>> X_train, X_test;
        std::vector<double> Y_train, Y_test;

        for (int i = 0; i < n; i++) {
            if (i < train_n) {
                X_train.push_back(X[idx[i]]);
                Y_train.push_back(Y[idx[i]]);
            } else {
                X_test.push_back(X[idx[i]]);
                Y_test.push_back(Y[idx[i]]);
            }
        }

        NormStats stats = compute_norm_stats(X_train);
        apply_norm_stats(X_train, stats);
        apply_norm_stats(X_test, stats);

        MLP model(d, {16, 1});

        auto start = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < epochs; epoch++) {
            ValuePtr total_loss = make_value(0.0);

            for (int i = 0; i < (int)X_train.size(); i++) {
                std::vector<ValuePtr> xvals;
                for (double x : X_train[i]) {
                    xvals.push_back(make_value(x));
                }

                auto pred = model(xvals)[0];
                auto target = make_value(Y_train[i]);

                auto diff = sub(pred, target);
                auto loss = square(diff);
                total_loss = add(total_loss, loss);
            }

            if (std::isnan(total_loss->data) || std::isinf(total_loss->data)) {
                throw std::runtime_error("Loss became NaN or Inf.");
            }

            double scale = 1.0 / X_train.size();
            auto params = model.parameters();

            for (auto& p : params) {
                p->grad = 0.0;
            }

            total_loss->backward();

            for (auto& p : params) {
                p->data -= lr * scale * p->grad;
            }

            if (epoch % 20 == 0 || epoch == epochs - 1) {
                double train_acc = compute_accuracy(model, X_train, Y_train);
                double test_acc = compute_accuracy(model, X_test, Y_test);

                std::cout << "Epoch " << epoch
                          << ", Loss = " << total_loss->data
                          << ", Train Acc = " << train_acc
                          << ", Test Acc = " << test_acc
                          << "\n";
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double train_time = std::chrono::duration<double>(end - start).count();

        double final_train_acc = compute_accuracy(model, X_train, Y_train);
        double final_test_acc = compute_accuracy(model, X_test, Y_test);

        std::cout << "\nFinal Train Accuracy: " << final_train_acc << "\n";
        std::cout << "Final Test Accuracy: " << final_test_acc << "\n";
        std::cout << "Training Time (s): " << train_time << "\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
