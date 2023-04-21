import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.simulateLoader import Stimulated256Input
from model.FISTA.DTVSTAnet import DTVNet
from model.FISTA.FISTAnet import FistaNet
from options import trainPath, inputTrainData, validPath, inputValidData, checkpointPath, debugPath, pretrain2
from loss import draw, drawLatent
from loss import perceptualLossCal as lossFunction

device = 4
validSet = Stimulated256Input("/home/nanovision/wyk/data/test", "/home/nanovision/wyk/data/testInput", device)
validLoader = DataLoader(validSet, batch_size=1, shuffle=False)

net = DTVNet((256,256,64),5).to(device)
# net = FistaNet(5).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
dictionary = torch.load(pretrain2)
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
            tic = time.time()
            output = net(input, projection)
            print("time:{}".format(time.time()-tic))
            loss = lossFunction(output[-1], label)
            draw(output, "/home/nanovision/wyk/data/testoutput", label)
            #drawLatent(ls, "/home/nanovision/wyk/data/testoutput")
            #for k,item in enumerate(output):
            #    item.detach().cpu().numpy().tofile("/home/nanovision/wyk/data/testoutput/{}.raw".format(k))
            iterator.set_postfix_str("loss:{}".format(loss.item()))
