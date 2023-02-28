import torch
import torch.nn as nn

class Dual(nn.Module):
    def __init__(self, features = 16, kernelSize = (3,3,3), paddingSize=(1,1,1)):
        super(Dual, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, features, kernelSize, padding=paddingSize),
            nn.ReLU(),
            nn.Conv3d(features, 1, kernelSize, padding=paddingSize),
            nn.ReLU(),
        )

    def forward(self, image, lamb):
        x = self.encoder(image)
        out = torch.sign(x) * nn.functional.relu(torch.abs(x) - nn.functional.relu(lamb))
        return self.decoder(out), nn.functional.l1_loss(x, torch.zeros_like(x))
