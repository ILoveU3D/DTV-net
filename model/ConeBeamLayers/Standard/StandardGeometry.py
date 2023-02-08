import numpy as np
import torch
import JITStandardGeometry as projector
from options import *

angles = torch.from_numpy(np.linspace(0, 360, standardAngleNum, False))
volumeSize = torch.tensor(standardVolumeSize)
detectorSize = torch.tensor(standardDetectorSize)

class ForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        device = input.device
        sino = projector.forward(input, angles.to(device), volumeSize.to(device),detectorSize.to(device), standardSID, standardSDD, device.index)
        return sino

    @staticmethod
    def backward(ctx, grad):
        device = grad.device
        volume = projector.backward(grad, angles.to(device), volumeSize.to(device), detectorSize.to(device), standardSID, standardSDD, device.index)
        return volume

class BackProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        device = input.device
        volume = projector.backward(input, angles.to(device), volumeSize.to(device), detectorSize.to(device), standardSID, standardSDD, device.index)
        return volume

    @staticmethod
    def backward(ctx, grad):
        device = grad.device
        sino = projector.forward(grad, angles.to(device), volumeSize.to(device), detectorSize.to(device), standardSID, standardSDD, device.index)
        return sino

class StandardGeometry(torch.nn.Module):
    def __init__(self):
        super(StandardGeometry, self).__init__()
        self.lamb = 1.0

    def forward(self, x, p):
        residual = ForwardProjection.apply(x) - p
        return x - self.lamb * BackProjection.apply(residual)

class StandardGeometryWithFBP(torch.nn.Module):
    def __init__(self):
        super(StandardGeometryWithFBP, self).__init__()
        self.lamb = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.cosWeight = torch.nn.Parameter(self.__cosWeight__(),requires_grad=False)
        # self.ramp = torch.nn.Parameter(self.__filter__(), requires_grad=False)
        self.ramp = torch.nn.Parameter(self.__conv__(detectorSize[0]), requires_grad=False)

    def forward(self, x, p):
        residual = ForwardProjection.apply(x) - p
        residual = residual * self.cosWeight
        # --* frequency domain filtering *--
        # residual = torch.fft.fft(residual, norm="forward")
        # residual = residual * self.ramp
        # residual = torch.fft.ifft(residual, norm="backward")
        # residual = torch.real(residual).contiguous()
        # --* space domain convolution *--
        b, c, a, h, w = residual.size()
        residual = residual.view(b, c, a * h, w)
        residual = torch.nn.functional.conv2d(residual, self.ramp, stride=1, padding=(0, int(detectorSize[0]/2)))
        residual = residual.view(b, c, a, h, w)
        return self.lamb * BackProjection.apply(residual)

    def __cosWeight__(self):
        cosine = np.zeros([1,1,standardAngleNum,detectorSize[1],detectorSize[0]], dtype=np.float32)
        mid = np.array(detectorSize) / 2
        for i in range(detectorSize[1]):
            for j in range(detectorSize[0]):
                cosine[...,i,j] = standardSDD / np.sqrt(standardSDD**2 + (i-mid[1])**2 + (j-mid[0])**2)
        return torch.from_numpy(cosine)

    def __filter__(self):
        filter = np.zeros([1, 1, standardAngleNum, detectorSize[1], detectorSize[0]], dtype=np.float32)
        mid = detectorSize[0] / 2
        for i in range(detectorSize[0]):
            filter[...,i] = mid - np.abs(i - mid)
        return torch.from_numpy(filter)

    def __conv__(self, projWidth):
        filter = np.ones([1,1,1,projWidth+1], dtype=np.float32)
        mid = np.floor(projWidth / 2)
        for i in range(projWidth+1):
            if (i - mid) % 2 == 0:
                filter[...,i] = 0
            else:
                filter[...,i] = -0.5 / (np.pi * np.pi * (i - mid) * (i - mid))
            if i == mid:
                filter[...,i] = 1 / 8
        return torch.from_numpy(filter)
