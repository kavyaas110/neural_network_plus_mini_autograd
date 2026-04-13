#include "data_utils.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>

std::pair<std::vector<std::vector<double>>, std::vector<double>>
read_training_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open training file: " + filename);
    }

    std::vector<std::vector<double>> X;
    std::vector<double> Y;

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (first_line) {
            first_line = false;
            if (!line.empty() && !(std::isdigit(line[0]) || line[0] == '-')) {
                continue;
            }
        }

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
    bool first_line = true;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (first_line) {
            first_line = false;
            if (!line.empty() && !(std::isdigit(line[0]) || line[0] == '-')) {
                continue;
            }
        }

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
