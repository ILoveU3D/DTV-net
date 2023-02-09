import torch
import numpy as np
from model.ConeBeamLayers.Beijing.BeijingGeometry import BeijingGeometryWithFBP, BeijingGeometry, ForwardProjection, BackProjection
from model.FISTA.DTVFISTA import DTVFista
from options import trainPath, outputPath

# data = np.fromfile(trainPath + "/pa_1.raw", dtype="float32")
data = np.fromfile("/media/wyk/wyk/Data/raws/SheppLogan.raw", dtype="float32")
data = np.reshape(data, [1,1,64,256,256])
data = torch.from_numpy(data).cuda()
projection = ForwardProjection.apply(data)
projection.detach().cpu().numpy().tofile(outputPath)
print("projected")
net = BeijingGeometryWithFBP().cuda().eval()
volume = net(torch.zeros_like(data), projection)
print("infered")

volume.detach().cpu().numpy().tofile(outputPath)