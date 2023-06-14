import os
import torch
import numpy as np
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection,BackProjection, BeijingGeometryWithFBP
from model.PDHG.ASDPOCS import ASDPOCS
from options import outputPath, trainPath

dtv = ASDPOCS((256,256,64), 700, False)
fdk = BeijingGeometryWithFBP().cuda()
for i in ["36.raw","126.raw","1064.raw"]:
    data = np.fromfile(os.path.join("/home/nanovision/wyk/data/trainDataCq500",i), dtype="float32")
    data = np.reshape(data, [1, 1, 64, 256, 256])
    data = torch.from_numpy(data).cuda()
    projection = ForwardProjection.apply(data)
    volume = torch.zeros_like(data).cuda()
    f = dtv.run(torch.zeros_like(data), projection)
    # f = fdk(volume, projection)
    f.detach().cpu().numpy().tofile(os.path.join("/home/nanovision/wyk/data/debug", i))
    print("infered {}".format(i))
