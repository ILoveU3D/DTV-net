import torch
import torch.nn as nn
from model.ConeBeamLayers.Standard.StandardGeometry import StandardGeometryWithFBP
from model.FISTA.RegularizationLayers.CNN import Dual
from options import debugPath

class FistaNet(nn.Module):
    def __init__(self, cascades:int=3):
        super(FistaNet, self).__init__()
        self.cascades = cascades
        self.ITE = nn.ModuleList([StandardGeometryWithFBP()] * cascades)
        self.AE = nn.ModuleList([Dual()] * cascades)
        self.sigma = nn.Parameter(torch.tensor([1.0] * cascades), requires_grad=True)

    def forward(self, image, sino):
        t = [0] * (self.cascades + 1)
        t[0] = image
        sparse = 0
        for cascade in range(self.cascades):
            z = self.ITE[cascade](t[cascade], sino)
            t[cascade + 1], sp = self.AE[cascade](z, self.sigma[cascade])
            sparse += sp
        return t, sparse