import torch
import torch.nn as nn
from model.ConeBeamLayers.Standard.StandardGeometry import ForwardProjection,BackProjection
from model.PDHG.RegularizationLayers.CNN import Dual

class LEARN(nn.Module):
    def __init__(self, cascades:int=30, debug:bool=False):
        super(LEARN, self).__init__()
        self.cascades = cascades
        self.debug = debug
        self.Dual = nn.ModuleList([Dual()] * cascades)
        self.lamb = nn.Parameter(torch.tensor([10e-4] * cascades), requires_grad=True)
        self.tao = nn.Parameter(torch.tensor([10e-4] * cascades), requires_grad=True)

    def forward(self, image, sino):
        t = [torch.tensor(0)] * (self.cascades + 1)
        t[0] = image
        for cascade in range(self.cascades):
            w = self.lamb[cascade] * BackProjection.apply(ForwardProjection.apply(t[cascade]) - sino)
            prox = self.tao[cascade] * self.Dual[cascade](t[cascade])
            t[cascade + 1] = t[cascade] - w - prox
        return t