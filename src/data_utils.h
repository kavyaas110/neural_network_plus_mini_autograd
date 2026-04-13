#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <string>
#include <vector>
#include <utility>

std::pair<std::vector<std::vector<double>>, std::vector<double>>
read_training_csv(const std::string& filename);

std::vector<std::vector<double>>
read_prediction_csv(const std::string& filename);

#endif
