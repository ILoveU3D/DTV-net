import os
import numpy as np
import torch
import scipy.io as sco
from torch.utils.data import Dataset
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection
import model.ConeBeamLayers.Beijing.BeijingGeometry as Geometry

class Real256Input(Dataset):
    def __init__(self, dirs, device, scale=(1,64,256,256), pScale=(1,144*21,128,80)):
        self.dir = []
        for dir in dirs:
            for item in os.listdir(os.path.join(dir, "label")):
                check = {}
                check["label"] = os.path.join(dir, "label", item)
                check["input"] = os.path.join(dir, "input", item)
                check["matlab"] = os.path.join(dir, "matlab", item.replace(".raw",".mat"))
                check["sino"] = os.path.join(dir, "sino", item)
                self.dir.append(check)
        self.scale = scale
        self.pScale = pScale
        self.device = device
        self.amp = 6e4

    def __getitem__(self,index):
        check = self.dir[index]
        img = np.fromfile(check["label"], dtype="float32")
        img = np.reshape(img, self.scale) * self.amp
        img = torch.unsqueeze(torch.from_numpy(img),0).to(self.device)
        img = torch.nn.functional.relu(img)
        proj = np.fromfile(check["sino"], dtype="float32")
        proj = np.reshape(proj, self.pScale) * self.amp
        proj = torch.unsqueeze(torch.from_numpy(proj),0).to(self.device)
        input = np.fromfile(check["input"], dtype="float32")
        input = np.reshape(input, self.scale) * self.amp
        input = torch.unsqueeze(torch.from_numpy(input),0).to(self.device)
        input = torch.nn.functional.relu(input)
        img = torch.squeeze(img, 0).cpu()
        input = torch.squeeze(input, 0).cpu()
        proj = torch.squeeze(proj, 0).cpu()
        parameters = sco.loadmat(check["matlab"])
        parameters = np.array(parameters["projection_matrix"]).astype(np.float32)
        parameters = torch.from_numpy(parameters).contiguous()
        Geometry.parameters = parameters
        return input, proj, img

    def __len__(self):
        return len(self.dir)

