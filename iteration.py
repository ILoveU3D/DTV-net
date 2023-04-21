import os
import tqdm
import torch
import argparse
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from model.ConeBeamLayers.Beijing.BeijingGeometry import BeijingGeometryWithFBP, BeijingGeometry, ForwardProjection, BackProjection
parser = argparse.ArgumentParser()

class IvrSet(Dataset):
    def __init__(self, root:str, s:int, device:int):
        files = os.listdir(root)
        start = int(np.ceil(len(files) / s) * device)
        end = int(np.ceil(len(files) / s) * (device + 1))
        if end > len(files): end = len(files)
        self.files = files[start:end]
        self.device = device
        self.root = root

    def __getitem__(self, item):
        data = np.fromfile(os.path.join(self.root, self.files[item]), dtype="float32")
        data = np.reshape(data, [1, 1, 64, 256, 256])
        data = torch.from_numpy(data).to(self.device)
        projection = ForwardProjection.apply(data)
        volume = torch.zeros_like(data).to(self.device)
        return volume.to(self.device), projection.to(self.device), self.files[item]

    def __len__(self):
        return len(self.files)

class Ivr(Module):
    def __init__(self, device:int):
        super().__init__()
        self.ivr = BeijingGeometry().to(device)

    def forward(self, volume, projection):
        for j in tqdm.trange(300):
            volume = volume + 3 * self.ivr(volume, projection)
        return volume


# sourcePath = "/media/wyk/wyk/Data/result/origin"
# targetPath = "/media/wyk/wyk/Data/result/360"
# sourcePath = "/media/wyk/wyk/Data/raws/trainData"
# targetPath = "/media/wyk/wyk/Data/raws/inputTrainData"

sourcePath = "/home/nanovision/wyk/data/validData"
targetPath = "/home/nanovision/wyk/data/inputValidData"
parser.add_argument("-d", "--device", type=int, default=0)
parser.add_argument("-s", "--sum", type=int, default=1)
args = parser.parse_args()
dataset = IvrSet(sourcePath, args.sum, args.device)
dataloader = DataLoader(dataset, batch_size=1)
mdl = Ivr(args.device).eval()
for i, data in enumerate(dataloader):
    volume, projection, files = data
    result = mdl(volume, projection)
    result.detach().cpu().numpy().tofile(os.path.join(targetPath, files[0]))
    print("infered {}".format(files[0]))

# net = BeijingGeometry().cuda().eval()
# for i in os.listdir(sourcePath):
#     data = np.fromfile(os.path.join(sourcePath,i), dtype="float32")
#     data = np.reshape(data, [1, 1, 64, 256, 256])
#     data = torch.from_numpy(data).cuda()
#     projection = ForwardProjection.apply(data)
#     volume = torch.zeros_like(data).cuda()
#     for j in tqdm.trange(200):
#         volume = volume + 0.2 * net(volume, projection)
#     volume.detach().cpu().numpy().tofile(os.path.join(targetPath, i))
#     print("infered {}".format(i))
