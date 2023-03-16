import torch
import torch.nn as nn
from model.ConeBeamLayers.Beijing.BeijingGeometry import BeijingGeometry
from model.FISTA.RegularizationLayers.BasicBlock import BasicBlock
from model.FISTA.RegularizationLayers.NEG import Neg
from options import debugPath

class FistaNet(nn.Module):
    def __init__(self, cascades:int=3):
        super(FistaNet, self).__init__()
        self.cascades = cascades
        self.ITE = nn.ModuleList([BeijingGeometry()] * cascades)
        self.AE = nn.ModuleList([BasicBlock()] * cascades)
        self.sigma = nn.Parameter(torch.tensor([1.0] * cascades), requires_grad=False)
        self.mu = nn.Parameter(torch.tensor([0.0] * cascades), requires_grad=True)

    def forward(self, image, sino):
        t = [0] * (self.cascades + 1)
        t[0] = image
        for cascade in range(self.cascades):
            z = t[cascade] + self.ITE[cascade](t[cascade], sino)
            z = self.AE[cascade](z)
            t[cascade + 1] = t[cascade] + self.mu[cascade] * (t[cascade] - z)
        return t
