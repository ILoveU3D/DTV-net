import os
import time
import tqdm
import torch
import numpy as np
from model.ConeBeamLayers.Shenzhen.ShenzhenGeometry import ShenzhenGeometryWithFBP, ForwardProjection, BackProjection
from model.FISTA.DTVFISTA import DTVFista
from options import trainPath, validPath, outputPath

# data = np.fromfile("/media/wyk/wyk/Data/raws/trainData/pa_1.raw", dtype="float32")
data = np.fromfile("/media/wyk/wyk/Data/raws/SheppLogan.raw", dtype="float32")
data = np.reshape(data, [1,1,64,256,256])
data = torch.from_numpy(data).cuda()
projection = ForwardProjection.apply(data)
# projection.detach().cpu().numpy().tofile("/media/wyk/wyk/Data/raws/r.raw")
print("projected")
net = ShenzhenGeometryWithFBP().cuda().eval()
volume = torch.zeros_like(data).cuda()
volume = net(volume, projection)
volume.detach().cpu().numpy().tofile("/media/wyk/wyk/Data/raws/r.raw")


# data = np.fromfile(os.path.join("/media/wyk/wyk/Data/result/origin",i), dtype="float32")
# data = np.reshape(data, [1, 1, 64, 256, 256])
# data = torch.from_numpy(data).cuda()
# projection = ForwardProjection.apply(data)
# volume = torch.zeros_like(data).cuda()
# for j in tqdm.trange(1000):
#     volume = volume + net(volume, projection)
# volume.detach().cpu().numpy().tofile(os.path.join("/media/wyk/wyk/Data/result/360", i))
# print("infered {}".format(i))