import os
import numpy as np
import torch
from torch.utils.data import Dataset
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection

class Stimulated256(Dataset):
    def __init__(self, dir, device, scale=(1,64,256,256)):
        self.dir = []
        for item in os.listdir(dir):
            self.dir.append(os.path.join(dir,item))
        self.scale = scale
        self.device = device

    def __getitem__(self,index):
        img = np.fromfile(self.dir[index], "float32")
        img = np.reshape(img, self.scale)
        img = torch.unsqueeze(torch.from_numpy(img),0).to(self.device)
        proj = ForwardProjection.apply(img)
        input = torch.zeros_like(img)
        img = torch.squeeze(img, 0).cpu()
        input = torch.squeeze(input, 0).cpu()
        proj = torch.squeeze(proj, 0).cpu()
        return input, proj, img

    def __len__(self):
        return len(self.dir)

class Stimulated256Input(Dataset):
    def __init__(self, dir, inputDir, device, scale=(1,64,256,256)):
        self.dir = []
        self.input = []
        for item in os.listdir(dir):
            self.dir.append(os.path.join(dir,item))
            self.input.append(os.path.join(inputDir,item))
        self.scale = scale
        self.device = device
        # self.weight = np.fromfile("/home/nanovision/wyk/data/weight.raw",dtype=np.float32)
        # self.weight = torch.from_numpy(self.weight.reshape(scale))

    def __getitem__(self,index):
        img = np.fromfile(self.dir[index], "float32")
        img = np.reshape(img, self.scale)
        img = torch.unsqueeze(torch.from_numpy(img),0).to(self.device)
        proj = ForwardProjection.apply(img)
        input = np.fromfile(self.input[index], "float32")
        input = np.reshape(input, self.scale)
        input = torch.unsqueeze(torch.from_numpy(input), 0).to(self.device)
        img = torch.squeeze(img, 0).cpu()
        input = torch.squeeze(input, 0).cpu()
        proj = torch.squeeze(proj, 0).cpu()
        return input, proj, img

    def __len__(self):
        return len(self.dir)

class Stimulated256InputShepp(Dataset):
    def __init__(self, dir, inputDir, scale=(1,64,256,256)):
        self.dir = []
        self.input = []
        for item in os.listdir(dir):
            self.dir.append(os.path.join(dir,item))
            self.input.append(os.path.join(inputDir,item))
        self.scale = scale

    def __getitem__(self,index):
        img = np.fromfile(self.dir[index], "float32")
        img = np.reshape(img, self.scale) * 1000
        img = torch.unsqueeze(torch.from_numpy(img),0).cuda()
        proj = ForwardProjection.apply(img)
        input = np.fromfile(self.input[index], "float32")
        input = np.reshape(input, self.scale) * 1000
        input = torch.unsqueeze(torch.from_numpy(input), 0).cuda()
        img *= self.weight.to(img.device)
        input *= self.weight.to(input.device)
        img = torch.squeeze(img, 0).cpu()
        input = torch.squeeze(input, 0).cpu()
        proj = torch.squeeze(proj, 0).cpu()
        return input, proj, img

    def __len__(self):
        return len(self.dir)
