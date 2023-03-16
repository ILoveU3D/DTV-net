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
        self.lamb = 0.1

    def run(self, image, sino):
        f = fp = image
        t = 1
        for cascade in range(self.cascades):
            z = f - 5e-3 * BackProjection.apply(ForwardProjection.apply(f)-sino)
            z = torch.nn.functional.relu(torch.abs(z) - 5e-4*self.lamb)
            tp = (1+np.sqrt(1+4*t**2))/2
            f = z + (t-1)/tp*(z-fp)
            fp = f
            t = tp
        return f