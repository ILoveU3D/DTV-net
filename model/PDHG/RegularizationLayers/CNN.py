import torch
import torch.nn as nn

class Dual(nn.Module):
    def __init__(self, features = 8, kernelSize = (3,3,3), paddingSize=(1,1,1)):
        super(Dual, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv3d(1, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, 1, kernelSize, padding=paddingSize),
        )

    def forward(self, image):
        return self.convolutions(image)