import torch
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection,BackProjection
from model.PDHG.Utils.Gradients import spraseMatrixX, spraseMatrixY, spraseMatrixZ, spraseMatrixI, getNormK
from model.PDHG.Utils.HyperParams import power_method
from model.PDHG.Utils.L1ball import l1ball

class DTVCP():
    def __init__(self, volumeSize:tuple, cascades:int=30, debug:bool=False):
        self.cascades = cascades
        self.debug = debug
        self.dx, self.dxt, normDx = spraseMatrixX(volumeSize)
        self.dy, self.dyt, normDy = spraseMatrixY(volumeSize)
        self.dz, self.dzt, normDz = spraseMatrixZ(volumeSize)
        self.dx, self.dxt = self.dx.cuda(), self.dxt.cuda()
        self.dy, self.dyt = self.dy.cuda(), self.dyt.cuda()
        self.dz, self.dzt = self.dz.cuda(), self.dzt.cuda()
        _, _, normI = spraseMatrixI(volumeSize)
        normH = 10e4
        self.v1 = normH / normDx
        self.v2 = normH / normDy
        self.v3 = normH / normDz
        self.mu = normH / normI
        L = 10e6
        print("||K||={}".format(L))
        b = 1
        self.tao = b/L
        self.sigma = 1/b/L

    def run(self, image, sino):
        w = torch.zeros_like(sino)
        p = q = s = c = torch.zeros_like(image)
        f = f_= image
        for cascade in range(self.cascades):
            if cascade % 10 == 0:
                self.tx = torch.sum(torch.abs(self.__getGradient(f, self.dx))).item()
                self.ty = torch.sum(torch.abs(self.__getGradient(f, self.dy))).item()
                self.tz = torch.sum(torch.abs(self.__getGradient(f, self.dz))).item()
            res = ForwardProjection.apply(f_)-sino
            w = (w + self.sigma * res)/(1+self.sigma)
            recon = BackProjection.apply(w)
            p_ = p + self.v1 * self.sigma * self.__getGradient(f_, self.dx)
            q_ = q + self.v2 * self.sigma * self.__getGradient(f_, self.dy)
            s_ = s + self.v3 * self.sigma * self.__getGradient(f_, self.dz)
            p = p_ - self.sigma * torch.sign(p_) * l1ball(torch.abs(p_) / self.sigma, self.v1*self.tx)
            q = q_ - self.sigma * torch.sign(q_) * l1ball(torch.abs(q_) / self.sigma, self.v2*self.ty)
            s = s_ - self.sigma * torch.sign(s_) * l1ball(torch.abs(s_) / self.sigma, self.v3*self.tz)
            c = -torch.nn.functional.relu(-c - self.sigma * self.mu * f_)
            p_ = self.v1 * self.__getGradient(p, self.dxt)
            q_ = self.v2 * self.__getGradient(q, self.dyt)
            s_ = self.v3 * self.__getGradient(s, self.dzt)
            c_ = self.mu * c
            fnew = f - recon - self.tao*(p_ + q_ + s_ + c_)
            f_ = 2 * fnew - f
            f = fnew
            loss = torch.sum(torch.abs(res))
            print("{}: loss is {:.2f}".format(cascade, loss))
            if self.debug and cascade % 100 == 0:
                f.detach().cpu().numpy().tofile(r"/media/seu/wyk/Recon/DTV/results/result_{}_{}.raw".format(cascade, loss))
                pass
        return f

    def __getGradient(self, image, sparse):
        result = []
        for batch in image:
            result.append(torch.reshape(torch.matmul(sparse,batch.view(-1)),batch.size()))
        return torch.stack(result, 0)
