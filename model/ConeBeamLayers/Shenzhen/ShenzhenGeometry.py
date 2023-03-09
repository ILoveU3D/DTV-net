import numpy as np
import torch
import scipy.io as sco
import JITShenzhenGeometry as projector
from options import *

parameters = sco.loadmat(shenzhenParameterRoot)
parameters = np.array(parameters["vec_all"]).astype(np.float32)
parameters = torch.from_numpy(parameters).contiguous()
volumeSize = torch.IntTensor(shenzhenVolumeSize)
detectorSize = torch.IntTensor(shenzhenDetectorSize)
class ForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        device = input.device
        sino = projector.forward(input, volumeSize.to(device), detectorSize.to(device), parameters.to(device), 0, 1, 1, device.index)
        return sino.reshape(1, shenzhenAngleNum, shenzhenDetectorSize[1], shenzhenDetectorSize[0]).permute(1,2,0,3).reshape(1, 1, shenzhenAngleNum, shenzhenDetectorSize[1], shenzhenDetectorSize[0])

    @staticmethod
    def backward(ctx, grad):
        device = grad.device
        grad = grad.reshape(shenzhenAngleNum, shenzhenDetectorSize[1], 1, shenzhenDetectorSize[0]).permute(2,0,1,3).reshape(1, 1, shenzhenAngleNum, shenzhenDetectorSize[1], shenzhenDetectorSize[0])
        volume = projector.backward(grad, volumeSize.to(device), detectorSize.to(device), parameters.to(device), 0, 1, 1, 905, -1774, 4018, 2115, device.index)
        return volume

class BackProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        device = input.device
        input = input.reshape(shenzhenAngleNum, shenzhenDetectorSize[1], 1, shenzhenDetectorSize[0]).permute(2,0,1,3).reshape(1, 1, shenzhenAngleNum, shenzhenDetectorSize[1], shenzhenDetectorSize[0])
        volume = projector.backward(input, volumeSize.to(device), detectorSize.to(device), parameters.to(device), 0, 1, 1, 905, -1774, 4018, 2115, device.index)
        return volume

    @staticmethod
    def backward(ctx, grad):
        device = grad.device
        sino = projector.forward(grad, volumeSize.to(device), detectorSize.to(device), parameters.to(device), 0, 1, 1, device.index)
        return sino.reshape(1, shenzhenAngleNum, shenzhenDetectorSize[1], shenzhenDetectorSize[0]).permute(1,2,0,3).reshape(1, 1, shenzhenAngleNum, shenzhenDetectorSize[1], shenzhenDetectorSize[0])

class CosWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        device = input.device
        sino = projector.cosweight(input, detectorSize.to(device), parameters.to(device), device.index)
        return sino.reshape(1, shenzhenAngleNum, shenzhenDetectorSize[1], shenzhenDetectorSize[0]).permute(1,2,0,3).reshape(1, 1, shenzhenAngleNum, shenzhenDetectorSize[1], shenzhenDetectorSize[0])

    @staticmethod
    def backward(ctx, grad):
        return grad

class ShenzhenGeometry(torch.nn.Module):
    def __init__(self):
        super(ShenzhenGeometry, self).__init__()
        self.lamb = torch.nn.Parameter(torch.tensor(10e-2), requires_grad=False)

    def forward(self, x, p):
        residual = ForwardProjection.apply(x) - p
        return x - self.lamb * BackProjection.apply(residual)

class ShenzhenGeometryWithFBP(torch.nn.Module):
    def __init__(self):
        super(ShenzhenGeometryWithFBP, self).__init__()
        self.lamb = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.ramp = torch.nn.Parameter(self.__conv__(shenzhenDetectorSize[0]), requires_grad=False)

    def forward(self, x, p):
        residual = ForwardProjection.apply(x) - p
        residual = residual.view(1, 1, shenzhenAngleNum, shenzhenDetectorSize[1], shenzhenDetectorSize[0])
        residual = CosWeight.apply(residual)
        residual = residual.view(1, 1, shenzhenAngleNum * shenzhenDetectorSize[1], shenzhenDetectorSize[0])
        residual = torch.nn.functional.conv2d(residual, self.ramp, stride = (1,1), padding = (0, int(shenzhenDetectorSize[0]/2)))
        residual = residual.view(1, 1, shenzhenAngleNum, shenzhenDetectorSize[1], shenzhenDetectorSize[0])
        return x - self.lamb * BackProjection.apply(residual)

    def __conv__(self, projWidth):
        filter = np.ones([1,1,1,projWidth], dtype=np.float32)
        mid = np.floor(projWidth / 2)
        for i in range(projWidth):
            if (i - mid) % 2 == 0:
                filter[...,i] = 0
            else:
                filter[...,i] = -0.5 / (np.pi * np.pi * (i - mid) * (i - mid))
            if i == mid:
                filter[...,i] = 1 / 8
        return torch.from_numpy(filter)
