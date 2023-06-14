import torch
from tqdm import trange
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection,BackProjection
from model.PDHG.Utils.Gradients import spraseMatrixX, spraseMatrixY, spraseMatrixZ, spraseMatrixI, getNormK
from model.PDHG.Utils.HyperParams import power_method
from model.PDHG.Utils.L1ball import l1ball

class ASDPOCS():
    def __init__(self, volumeSize:tuple, cascades:int=30, debug:bool=False):
        self.cascades = cascades
        self.debug = debug
        self.dx, self.dxt, _ = spraseMatrixX(volumeSize)
        self.dy, self.dyt, _ = spraseMatrixY(volumeSize)
        self.dz, self.dzt, _ = spraseMatrixZ(volumeSize)
        self.dx = self.dx.cuda()
        self.dy = self.dy.cuda()
        self.dz = self.dz.cuda()
        self.dxt = self.dxt.cuda()
        self.dyt = self.dyt.cuda()
        self.dzt = self.dzt.cuda()
        self.beta = 0.00005
        self.beta_red = 0.9999
        self.ng = 10
        self.alpha = 0.001
        self.rmax = 0.95
        self.alpha_red = 0.95

    def run(self, image, sino):
        f = image
        f0 = image
        davg = 1
        for i in trange(self.cascades):
            f = f - self.beta * BackProjection.apply(ForwardProjection.apply(f) - sino)
            dp = torch.nn.functional.l1_loss(f0, f)
            if i == 0: davg = self.alpha * dp.item()
            f0 = f
            for j in range(self.ng):
                dx = self.__getGradient(f, self.dx)
                dy = self.__getGradient(f, self.dy)
                dz = self.__getGradient(f, self.dz)
                gradTVS = torch.sqrt(dx**2+dy**2+dz**2)+1
                gradTV = self.__getGradient(dx/gradTVS,self.dxt)+self.__getGradient(dy/gradTVS,self.dyt)+self.__getGradient(dz/gradTVS,self.dzt)
                f -= davg * gradTV
            dg = torch.nn.functional.l1_loss(f, f0)
            if dg > self.rmax * dp: davg *= self.alpha
            self.beta *= self.beta_red
        return torch.nn.functional.relu(f)

    def __getGradient(self, image, sparse):
        result = []
        for batch in image:
            result.append(torch.reshape(torch.matmul(sparse,batch.view(-1)),batch.size()))
        return torch.stack(result, 0)
