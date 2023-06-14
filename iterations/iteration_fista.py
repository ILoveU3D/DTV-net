import os
import torch
import numpy as np
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection,BackProjection, BeijingGeometryWithFBP
from model.FISTA.FISTA import DTVFista
from options import outputPath, trainPath

dtv = DTVFista(400, False)
fdk = BeijingGeometryWithFBP().cuda()
for i in os.listdir("/home/nanovision/wyk/data/result/AAPM/origin"):
    data = np.fromfile(os.path.join("/home/nanovision/wyk/data/result/AAPM/origin",i), dtype="float32")
    data = np.reshape(data, [1, 1, 64, 256, 256])
    data = torch.from_numpy(data).cuda()
    import scipy.io as sco
    import model.ConeBeamLayers.Beijing.BeijingGeometry as Geometry
    parameters = sco.loadmat("/home/nanovision/wyk/data/real/test/matlab/1_1.mat")
    parameters = np.array(parameters["projection_matrix"]).astype(np.float32)
    parameters = torch.from_numpy(parameters).contiguous()
    Geometry.parameters = parameters
    projection = ForwardProjection.apply(data)
    projection.detach().cpu().numpy().tofile(os.path.join("/home/nanovision/wyk/data/result/AAPM/FISTA", i))
    volume = torch.zeros_like(data).cuda()
    f = dtv.run(torch.zeros_like(data), projection)
    # f = fdk(volume, projection)
    f.detach().cpu().numpy().tofile(os.path.join("/home/nanovision/wyk/data/result/AAPM/FISTA", i))
    print("infered {}".format(i))
