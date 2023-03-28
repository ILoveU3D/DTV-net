import torch.nn as nn
import torch
import numpy as np

def _downsampler(x):
    return nn.functional.interpolate(x[..., 0:128, :, :], [64, 512, 512])

def _upsampler(x):
    xr = nn.functional.interpolate(x, [128, 512, 512])
    xr = nn.functional.pad(xr, (0, 0, 0, 0, 0, 384-128))
    return xr

if __name__ == '__main__':
    # data = np.fromfile("/media/wyk/wyk/Data/raws/trainData/pa_1.raw", dtype="float32")
    data = np.fromfile("/media/wyk/wyk/Data/raws/label.raw", dtype="float32")
    data = np.reshape(data, [1,1,384,512,512])
    data = torch.from_numpy(data).cuda()
    volume = _downsampler(data)
    volume = _upsampler(volume)
    volume.detach().cpu().numpy().tofile("/media/wyk/wyk/Data/raws/r.raw")