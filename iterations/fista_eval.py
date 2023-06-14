import os
import torch
import numpy as np
import scipy.io as sco
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection,BackProjection, BeijingGeometryWithFBP
from model.FISTA.FISTA import DTVFista

dtv = DTVFista(200, False)
projection = np.fromfile("/home/nanovision/wyk/data/real/projection_1.raw", dtype="float32")
projection = np.reshape(projection, [1080,21, 128, 80]) * 6e4
parameters = sco.loadmat("/home/nanovision/wyk/data/real/projVecReal_1.mat")
parameters = np.array(parameters["projection_matrix"]).astype(np.float32).reshape([1080,21,12])
projection_new = np.zeros((144,21,128,80)).astype(np.float32)
parameters_new = np.zeros((144,21,12)).astype(np.float32)
for j in range(24):
    projection_new[j*6:j*6+6] = projection[j*45:(j+1)*45:45//6+1]
    parameters_new[j*6:j*6+6] = parameters[j*45:(j+1)*45:45//6+1]
projection_new = np.reshape(projection_new, [1, 1, 144*21, 128, 80])
projection = torch.from_numpy(projection_new).cuda()
parameters_new = np.reshape(parameters_new, [144*21, 12])
parameters_new = torch.from_numpy(parameters_new).cuda().contiguous()
import model.ConeBeamLayers.Beijing.BeijingGeometry as Geometry
Geometry.parameters = parameters_new
volume = torch.zeros([1,1,64,256,256]).float().cuda()
f = dtv.run(volume, projection)
f.detach().cpu().numpy().tofile(os.path.join("/home/nanovision/wyk/data/result/real/FISTA/1_1.raw"))
print("infered")
