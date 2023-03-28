import torch
import numpy as np
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection,BackProjection
from model.PDHG.Utils.Gradients import spraseMatrixX, spraseMatrixY, spraseMatrixZ
from model.PDHG.Utils.HyperParams import power_method
from model.PDHG.Utils.L1ball import l1ball

class DTVFista():
    def __init__(self, volumeSize:tuple, cascades:int=30, debug:bool=False):
        self.cascades = cascades
        self.debug = debug
        self.lamb = 1e-2

    def run(self, image, sino):
        t = 1
        I = Ip = image
        y = I
        for cascade in range(self.cascades):
            d = y - 5e-3 * BackProjection.apply(ForwardProjection.apply(y)-sino)
            I = torch.sgn(d) * torch.nn.functional.relu(torch.abs(d) - self.lamb*5e-3)
            tp = (1+np.sqrt(1+4*t**2))/2
            y = I + (t-1)/tp * (I-Ip)
            Ip = I
            t = tp
        return I