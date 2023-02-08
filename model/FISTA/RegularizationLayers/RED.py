import torch
import torch.nn as nn
from model.ConeBeamLayers.Standard.StandardGeometry import StandardGeometryWithFBP

class Red(nn.Module):
    def __init__(self, features = 32, kernelSize = (3,3,3), paddingSize=(1,1,1)):
        super(Red, self).__init__()
        self.fbp = StandardGeometryWithFBP()
        self.encoder1 = nn.Sequential(
            nn.Conv3d(1, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
        )
        self.decoder2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
        )
        self.decoder3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, 1, kernelSize, padding=paddingSize),
        )
        self.decoder4 = nn.Sequential(
            nn.ReLU(),
        )

    def forward(self, image, proj):
        image = self.fbp(image, proj)
        x1 = self.encoder1(image)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.decoder1(x3) + x2
        x5 = self.decoder2(x4) + x1
        x6 = self.decoder3(x5) + image
        return self.decoder4(x6)

    def latent(self, image):
        x1 = self.encoder1(image)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        return torch.sum(x3.view(-1))