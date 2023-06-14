import os
import torch
import numpy as np
import scipy.io as sco
import model.ConeBeamLayers.Beijing.BeijingGeometry as Geometry
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection,BackProjection, BeijingGeometryWithFBP
from model.PDHG.DTVCP import DTVCP

dtv = DTVCP((256,256,64),900, False)
for i in os.listdir("/home/nanovision/wyk/data/real/test/sino"):
    projection = np.fromfile(os.path.join("/home/nanovision/wyk/data/real/test/sino",i), dtype="float32")
    projection = np.reshape(projection, [1, 1, 144*21, 128, 80]) * 6e4
    projection = torch.from_numpy(projection).cuda()
    parameters = sco.loadmat(os.path.join("/home/nanovision/wyk/data/real/test/matlab",i.replace(".raw",".mat")))
    parameters = np.array(parameters["projection_matrix"]).astype(np.float32)
    parameters = torch.from_numpy(parameters).contiguous()
    Geometry.parameters = parameters
    volume = torch.zeros([1,1,64,256,256]).float().cuda()
    f = dtv.run(volume, projection)
    f.detach().cpu().numpy().tofile(os.path.join("/home/nanovision/wyk/data/result/real/DTVCP", i))
    print("infered {}".format(i))
