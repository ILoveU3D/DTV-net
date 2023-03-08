import os
import time
import tqdm
import torch
import numpy as np
from model.ConeBeamLayers.Beijing.BeijingGeometry import BeijingGeometryWithFBP, BeijingGeometry, ForwardProjection, BackProjection
from model.FISTA.DTVFISTA import DTVFista
from options import trainPath, validPath, outputPath

# data = np.fromfile(trainPath + "/pa_1.raw", dtype="float32")
# data = np.fromfile("/media/wyk/wyk/Data/raws/SheppLogan.raw", dtype="float32")
# data = np.reshape(data, [1,1,64,256,256])
# data = torch.from_numpy(data).cuda()
# projection = ForwardProjection.apply(data)
# projection.detach().cpu().numpy().tofile(outputPath)
# print("projected")
net = BeijingGeometry().cuda().eval()
# volume = torch.zeros_like(data).cuda()
# volume = net(volume, projection)

data = np.fromfile("/media/wyk/wyk/Data/raws/SheppLogan.raw", dtype="float32")
data = np.reshape(data, [1, 1, 64, 256, 256])
data = torch.from_numpy(data).cuda()
projection = ForwardProjection.apply(data)
volume = torch.zeros_like(data).cuda()
for i in tqdm.trange(100):
    volume = net(volume, projection)
volume.detach().cpu().numpy().tofile("/media/wyk/wyk/Data/raws/SheppLoganInput.raw")
print("infered")