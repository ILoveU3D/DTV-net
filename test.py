import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.simulateLoader import Stimulated256Input
from model.FISTA.DTVSTAnetv2 import DTVNet
from model.FISTA.RegularizationLayers.RED import Red
from options import trainPath, inputTrainData, validPath, inputValidData, checkpointPath, debugPath, pretrain3
from loss import draw
from loss import stepLoss as lossFunction

validSet = Stimulated256Input("/home/nanovision/wyk/data/test", "/home/nanovision/wyk/data/testInput")
validLoader = DataLoader(validSet, batch_size=1, shuffle=False)

device = 4
net = DTVNet((256,256,64),5).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
dictionary = torch.load(pretrain3)
net.load_state_dict(dictionary["model"])
optimizer.load_state_dict(dictionary["optimizer"])
scheduler.load_state_dict(dictionary["scheduler"])

net.eval()
with torch.no_grad():
    with tqdm(validLoader) as iterator:
        iterator.set_description("validating...")
        for idx, data in enumerate(iterator):
            input, projection, label = data
            input, projection, label = input.to(device), projection.to(device), label.to(device)
            output = net(input, projection)
            loss = lossFunction(output, label)
            draw(output, "/home/nanovision/wyk/data/testoutput", label)
            iterator.set_postfix_str("loss:{}".format(loss.item()))
