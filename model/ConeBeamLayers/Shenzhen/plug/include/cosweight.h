#include <torch/extension.h>

torch::Tensor cosweight(torch::Tensor sino, torch::Tensor _detectorSize, torch::Tensor projectVector, const long device);