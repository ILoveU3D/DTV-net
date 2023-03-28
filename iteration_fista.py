import os
import torch
import numpy as np
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection,BackProjection
from model.FISTA.FISTA import DTVFista
from options import outputPath, trainPath

dtv = DTVFista((256, 256, 64), 900, False)
for i in os.listdir("/media/wyk/wyk/Data/result/origin"):
    data = np.fromfile(os.path.join("/media/wyk/wyk/Data/result/origin",i), dtype="float32")
    data = np.reshape(data, [1, 1, 64, 256, 256])
    data = torch.from_numpy(data).cuda()
    projection = ForwardProjection.apply(data)
    volume = torch.zeros_like(data).cuda()
    f = dtv.run(torch.zeros_like(data), projection)
    f.detach().cpu().numpy().tofile(os.path.join("/media/wyk/wyk/Data/result/FISTA", i))
    print("infered {}".format(i))