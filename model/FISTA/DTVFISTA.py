import torch
import numpy as np
from model.ConeBeamLayers.Standard.StandardGeometry import ForwardProjection,BackProjection
from model.PDHG.Utils.Gradients import spraseMatrixX, spraseMatrixY, spraseMatrixZ
from model.PDHG.Utils.HyperParams import power_method
from model.PDHG.Utils.L1ball import l1ball

class DTVFista():
    def __init__(self, volumeSize:tuple, cascades:int=30, debug:bool=False):
        self.cascades = cascades
        self.debug = debug
        self.dx, self.dxt, normDx = spraseMatrixX(volumeSize)
        self.dy, self.dyt, normDy = spraseMatrixY(volumeSize)
        self.dz, self.dzt, normDz = spraseMatrixZ(volumeSize)
        self.sigma = 1e-3
        self.v1 = normDx / 1e-3 * 0.88
        self.v2 = normDy / 1e-3 * 0.88
        self.v3 = normDz / 1e-3 * 0.88
        self.t1 = 1e-3
        self.t2 = 1e-3
        self.t3 = 1e-3

    def run(self, image, sino):
        f = image
        t = 1
        p = q = s = 0
        for cascade in range(self.cascades):
            tp = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            z = f - self.sigma * BackProjection.apply(ForwardProjection.apply(f)-sino)
            z = torch.nn.functional.relu(z)
            pnew = self.getGradient(z, self.dx)
            qnew = self.getGradient(z, self.dy)
            snew = self.getGradient(z, self.dz)
            p = torch.sign(pnew) * torch.max(torch.abs(pnew)-self.v1)
            q = torch.sign(qnew) * torch.max(torch.abs(qnew)-self.v2)
            s = torch.sign(snew) * torch.max(torch.abs(snew)-self.v3)
            fnew = self.t1*self.getGradient(p, self.dxt)+self.t2*self.getGradient(q, self.dyt)+self.t3*self.getGradient(s, self.dzt)+z
            f = fnew + (t-1)/tp*(fnew-f)
            print("{}: loss is {:.2f}, gradient l1:{}".format(cascade,torch.sum(torch.abs(ForwardProjection.apply(f)-sino)), torch.sum(torch.abs(p))))
            if self.debug and cascade % 50 == 0:
                f.detach().cpu().numpy().tofile(r"/media/seu/wyk/Recon/DTV/results/result_FDPG_{}_{}.raw".format(cascade, torch.sum(torch.abs(ForwardProjection.apply(f)-sino))))
                p.detach().cpu().numpy().tofile(r"/media/seu/wyk/Recon/DTV/results/p_{}.raw".format(cascade))
        return f

    def getGradient(self, image, sparse):
        result = []
        for batch in image:
            result.append(torch.reshape(torch.matmul(sparse,batch.view(-1)),batch.size()))
        return torch.stack(result, 0)