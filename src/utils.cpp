#include "utils.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

std::pair<std::vector<std::vector<double>>, std::vector<double>>
read_training_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open training file: " + filename);
    }

    std::vector<std::vector<double>> X;
    std::vector<double> Y;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }

        if (row.size() < 2) {
            throw std::runtime_error("Each training row must have at least 1 feature and 1 target.");
        }

        std::vector<double> features(row.begin(), row.end() - 1);
        double target = row.back();

        X.push_back(features);
        Y.push_back(target);
    }

    if (X.empty()) {
        throw std::runtime_error("Training file is empty.");
    }

    return {X, Y};
}

std::vector<std::vector<double>>
read_prediction_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open prediction file: " + filename);
    }

    std::vector<std::vector<double>> X;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }

        if (row.empty()) {
            throw std::runtime_error("Prediction row cannot be empty.");
        }

        X.push_back(row);
    }

    if (X.empty()) {
        throw std::runtime_error("Prediction file is empty.");
    }

    return X;
}

void save_model(const MLP& model, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open model file for writing: " + filename);
    }

    out << model.nin << "\n";
    out << model.nouts.size() << "\n";
    for (int n : model.nouts) {
        out << n << " ";
    }
    out << "\n";

    for (const auto& layer : model.layers) {
        for (const auto& neuron : layer.neurons) {
            for (const auto& weight : neuron.w) {
                out << weight->data << " ";
            }
            out << neuron.b->data << "\n";
        }
    }
}

MLP load_model(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open model file for reading: " + filename);
    }

    int nin;
    in >> nin;

    int num_layers;
    in >> num_layers;

    std::vector<int> nouts(num_layers);
    for (int i = 0; i < num_layers; i++) {
        in >> nouts[i];
    }

    MLP model(nin, nouts);

    for (auto& layer : model.layers) {
        for (auto& neuron : layer.neurons) {
            for (auto& weight : neuron.w) {
                in >> weight->data;
            }
            in >> neuron.b->data;
        }
    }

    return model;
}
