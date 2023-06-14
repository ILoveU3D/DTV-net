import torch
import numpy as np
from tqdm import trange
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection,BackProjection
from model.PDHG.Utils.Gradients import spraseMatrixX, spraseMatrixY, spraseMatrixZ
from model.PDHG.Utils.HyperParams import power_method
from model.PDHG.Utils.L1ball import l1ball

class DTVFista():
    def __init__(self, cascades:int=30, debug:bool=False):
        self.cascades = cascades
        self.debug = debug
        self.lamb = 10e3
        self.L = 0.0001

    def run(self, image, sino):
        t = 1
        I = Ip = image
        y = I
        for cascade in trange(self.cascades):
            d = y - self.L * BackProjection.apply(ForwardProjection.apply(y)-sino)
            I = torch.sign(d) * torch.nn.functional.relu(torch.abs(d) - self.lamb*self.L)
            tp = (1+np.sqrt(1+4*t**2))/2
            y = I + (t-1)/tp * (I-Ip)
            Ip = I
            t = tp
        return I
