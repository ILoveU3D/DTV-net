import torch
import torch.nn as nn
import numpy as np
from model.ConeBeamLayers.Beijing.BeijingGeometry import BeijingGeometry
from model.PDHG.Utils.Gradients import spraseMatrixX, spraseMatrixY, spraseMatrixZ
from model.FISTA.RegularizationLayers.BasicBlock import BasicBlock
from model.FISTA.RegularizationLayers.NEG import Neg

class DTVNet(nn.Module):
    def __init__(self, volumeSize, cascades: int = 3):
        super(DTVNet, self).__init__()
        self.cascades = cascades
        self.ITE = BeijingGeometry()
        self.AE = nn.ModuleList([BasicBlock(), BasicBlock(), BasicBlock(), BasicBlock()])
        # self.NG = Neg(32)
        self.dx, self.dxt, normDx = spraseMatrixX(volumeSize)
        self.dx, self.dxt = nn.Parameter(self.dx, requires_grad=False), nn.Parameter(self.dxt, requires_grad=False)
        self.dy, self.dyt, normDy = spraseMatrixY(volumeSize)
        self.dy, self.dyt = nn.Parameter(self.dy, requires_grad=False), nn.Parameter(self.dyt, requires_grad=False)
        self.dz, self.dzt, normDz = spraseMatrixZ(volumeSize)
        self.dz, self.dzt = nn.Parameter(self.dz, requires_grad=False), nn.Parameter(self.dzt, requires_grad=False)
        # self.sigma = nn.Parameter(torch.tensor([1.0] * 4), requires_grad=False)
        self.ntx = nn.Parameter(torch.tensor([0.0] * cascades), requires_grad=True)
        self.nty = nn.Parameter(torch.tensor([0.0] * cascades), requires_grad=True)
        self.ntz = nn.Parameter(torch.tensor([0.0] * cascades), requires_grad=True)
        self.nt = nn.Parameter(torch.tensor([0.0] * cascades), requires_grad=True)

    def forward(self, image, sino):
        t = [torch.tensor(0)] * (self.cascades + 1)
        t[0] = image
        p = q = s = 0
        for cascade in range(self.cascades):
            z = t[cascade] + self.ITE(t[cascade], sino)
            pnew = self.__getGradient(z, self.dx)
            qnew = self.__getGradient(z, self.dy)
            snew = self.__getGradient(z, self.dz)
            pnew = self.AE[0](pnew)
            qnew = self.AE[1](qnew)
            snew = self.AE[2](snew)
            znew = self.AE[3](z)
            p = p + self.ntx[cascade] * (p - pnew)
            q = q + self.nty[cascade] * (q - qnew)
            s = s + self.ntz[cascade] * (s - snew)
            z_ = t[cascade] + self.nt[cascade] * (t[cascade] - znew)
            p_ = self.__getGradient(p, self.dxt) 
            q_ = self.__getGradient(q, self.dyt)
            s_ = self.__getGradient(s, self.dzt)
            t[cascade+1] = p_ + q_ + s_ + z_
            # t[cascade+1] = t[cascade] + self.nt[cascade] * (t[cascade] - zp)
        return t

    def __getGradient(self, image, sparse):
        result = []
        for batch in image:
            result.append(torch.reshape(torch.matmul(sparse, batch.view(-1)), batch.size()))
        return torch.stack(result, 0)
