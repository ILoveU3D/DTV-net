import torch
import torch.nn as nn

class Neg(nn.Module):
    def __init__(self, features = 8, kernelSize = (3,3,3), paddingSize=(1,1,1), inputChannel=1, outputChannel=1):
        super(Neg, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv3d(inputChannel, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, outputChannel, kernelSize, padding=paddingSize),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(outputChannel, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, outputChannel, kernelSize, padding=paddingSize),
            nn.ReLU(),
        )

    def forward(self, image):
        x1 = image + self.encoder1(image)
        return x1 + self.encoder2(x1)
