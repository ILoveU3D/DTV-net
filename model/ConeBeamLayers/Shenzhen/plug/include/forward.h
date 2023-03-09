#include <torch/extension.h>

torch::Tensor forward(torch::Tensor volume, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector,
                      const float volbiasz, const float dSampleInterval, const float dSliceInterval, const long device);