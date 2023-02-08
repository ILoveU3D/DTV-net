import torch
import torch.nn as nn

class Neg(nn.Module):
    def __init__(self, features = 8, kernelSize = (3,3,3), paddingSize=(1,1,1)):
        super(Neg, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, 1, kernelSize, padding=paddingSize),
            nn.ReLU(),
        )

    def forward(self, image):
        return image + self.encoder(image)