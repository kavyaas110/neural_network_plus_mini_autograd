#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <utility>
#include "nn.h"

std::pair<std::vector<std::vector<double>>, std::vector<double>>
read_training_csv(const std::string& filename);

std::vector<std::vector<double>>
read_prediction_csv(const std::string& filename);

void save_model(const MLP& model, const std::string& filename);
MLP load_model(const std::string& filename);

#endif
