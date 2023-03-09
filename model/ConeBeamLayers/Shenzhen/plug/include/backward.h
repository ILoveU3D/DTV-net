#include <torch/extension.h>

torch::Tensor backward(torch::Tensor sino, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector,
                       const float volbiasz, const float dSampleInterval, const float dSliceInterval,
                       const float sourceRadius, const float sourceZpos, const float fBiaz, const float  SID,
                       const long device);