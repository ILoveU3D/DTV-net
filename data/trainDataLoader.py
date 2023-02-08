import os
import numpy as np
import torch
from torch.utils.data import Dataset
from model.ConeBeamLayers.Standard.StandardGeometry import ForwardProjection

class Stimulated256(Dataset):
    def __init__(self, dir, scale=(1,64,256,256)):
        self.dir = []
        for item in os.listdir(dir):
            self.dir.append(os.path.join(dir,item))
        self.scale = scale

    def __getitem__(self,index):
        img = np.fromfile(self.dir[index], "float32")
        img = np.reshape(img, self.scale)
        img = torch.unsqueeze(torch.from_numpy(img),0).cuda()
        proj = ForwardProjection.apply(img)
        input = torch.zeros_like(img)
        img = torch.squeeze(img, 0).cpu()
        input = torch.squeeze(input, 0).cpu()
        proj = torch.squeeze(proj, 0).cpu()
        return input, proj, img

    def __len__(self):
        return len(self.dir)