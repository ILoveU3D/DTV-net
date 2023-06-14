import os
import torch
import numpy as np
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection,BackProjection
from model.PDHG.DTVCP import DTVCP
from options import outputPath, trainPath

dtv = DTVCP((256, 256, 64), 501, False)
for i in os.listdir("/home/nanovision/wyk/data/result/Cq500/origin"):
    data = np.fromfile(os.path.join("/home/nanovision/wyk/data/result/Cq500/origin",i), dtype="float32")
    data = np.reshape(data, [1, 1, 64, 256, 256])
    data = torch.from_numpy(data).cuda()
    projection = ForwardProjection.apply(data)
    volume = torch.zeros_like(data).cuda()
    f = dtv.run(torch.zeros_like(data), projection)
    f.detach().cpu().numpy().tofile(os.path.join("/home/nanovision/wyk/data/result/Cq500/DTVCP", i))
    print("infered {}".format(i))
