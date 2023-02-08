import torch
import numpy as np
from model.ConeBeamLayers.Standard.StandardGeometry import ForwardProjection,BackProjection
from model.PDHG.DTVCP import DTVCP
from model.FISTA.DTVFISTA import DTVFista
from options import outputPath, trainPath

data = np.fromfile(trainPath+"/1.raw", dtype="float32")
data = np.reshape(data, [1,1,16,256,256])
data = torch.from_numpy(data).cuda()
projection = ForwardProjection.apply(data)
f = BackProjection.apply(projection)
dtv = DTVFista((256,256,16),501,True)
f = dtv.run(torch.zeros_like(data), projection)

