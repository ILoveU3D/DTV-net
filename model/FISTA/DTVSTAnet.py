import torch
import torch.nn as nn
import numpy as np
from model.ConeBeamLayers.Beijing.BeijingGeometry import BeijingGeometry
from model.PDHG.Utils.Gradients import spraseMatrixX, spraseMatrixY, spraseMatrixZ
from model.FISTA.RegularizationLayers.CNN import Dual

class DTVNet(nn.Module):
    def __init__(self, volumeSize, cascades: int = 3):
        super(DTVNet, self).__init__()
        self.cascades = cascades
        self.ITE = BeijingGeometry()
        self.AE = nn.ModuleList([Dual()] * 4)
        self.dx, self.dxt, normDx = spraseMatrixX(volumeSize)
        self.dx, self.dxt = nn.Parameter(self.dx, requires_grad=False), nn.Parameter(self.dxt, requires_grad=False)
        self.dy, self.dyt, normDy = spraseMatrixY(volumeSize)
        self.dy, self.dyt = nn.Parameter(self.dy, requires_grad=False), nn.Parameter(self.dyt, requires_grad=False)
        self.dz, self.dzt, normDz = spraseMatrixZ(volumeSize)
        self.dz, self.dzt = nn.Parameter(self.dz, requires_grad=False), nn.Parameter(self.dzt, requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor([1.0] * 4), requires_grad=True)
        self.nt = nn.Parameter(torch.tensor([0.0] * cascades), requires_grad=True)
        self.lamb = 0.01

    def forward(self, image, sino):
        t = [torch.tensor(0)] * (self.cascades + 1)
        t[0] = image
        p = q = s = 0
        for cascade in range(self.cascades):
            z = self.ITE(t[cascade], sino)
            pnew = p - self.__getGradient(z, self.dx)
            qnew = q - self.__getGradient(z, self.dy)
            snew = s - self.__getGradient(z, self.dz)
            pnew, _ = self.AE[0](pnew, self.sigma[0])
            qnew, _ = self.AE[1](qnew, self.sigma[1])
            snew, _ = self.AE[2](snew, self.sigma[2])
            znew, _ = self.AE[3](z, self.sigma[3])
            p = pnew + self.nt[cascade] * (pnew - p)
            q = qnew + self.nt[cascade] * (qnew - q)
            s = snew + self.nt[cascade] * (snew - s)
            p_ = self.__getGradient(p, self.dxt)
            q_ = self.__getGradient(q, self.dyt)
            s_ = self.__getGradient(s, self.dzt)
            t[cascade + 1] = p_ + q_ + s_ + znew
        return t

    def __getGradient(self, image, sparse):
        result = []
        for batch in image:
            result.append(torch.reshape(torch.matmul(sparse, batch.view(-1)), batch.size()))
        return torch.stack(result, 0)