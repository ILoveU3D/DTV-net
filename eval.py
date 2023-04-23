import os
import time
import tqdm
import torch
import numpy as np
# from model.ConeBeamLayers.Shenzhen.ShenzhenGeometry import ShenzhenGeometryWithFBP, ForwardProjection, BackProjection
from model.ConeBeamLayers.Beijing.BeijingGeometry import BeijingGeometry, BeijingGeometryWithFBP, ForwardProjection, BackProjection
from options import trainPath, validPath, outputPath

data = np.fromfile("/media/wyk/wyk/Data/raws/trainData/pa_1.raw", dtype="float32")
# data = np.fromfile("/media/wyk/wyk/Data/raws/volume.raw", dtype="float32")
data = np.reshape(data, [1,1,64,256,256])
data = torch.from_numpy(data).cuda()
projection = ForwardProjection.apply(data)
projection.detach().cpu().numpy().tofile("/media/wyk/wyk/Data/raws/sino2.raw")
print("projected")
# net = ShenzhenGeometryWithFBP().cuda().eval()
net = BeijingGeometryWithFBP().cuda().eval()
volume = torch.zeros_like(data).cuda()
volume = net(volume, projection)
volume.detach().cpu().numpy().tofile("/media/wyk/wyk/Data/raws/r.raw")