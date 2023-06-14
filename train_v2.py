import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.simulateLoader import Stimulated256Input
from model.FISTA.DTVSTAnet import DTVNet
from options import trainPath, inputTrainData, validPath, inputValidData, checkpointPath, debugPath, pretrain3
from loss import draw
from loss import stepLoss as lossFunction
debugPath = "/home/nanovision/wyk/data/train3"
lossFunction = torch.nn.L1Loss(reduction="mean")

device = 4
trainSet = Stimulated256Input(trainPath, inputTrainData, device)
validSet = Stimulated256Input(trainPath, inputTrainData, device)
trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True)
validLoader = DataLoader(validSet, batch_size=1, shuffle=False)

net = DTVNet((256,256,64),5).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
# dictionary = torch.load(pretrain3)
# net.load_state_dict(dictionary["model"])
# optimizer.load_state_dict(dictionary["optimizer"])
# scheduler.load_state_dict(dictionary["scheduler"])

epoch = 500

for i in range(epoch):
    trainLoss = []
    validLoss = []
    net.train()
    with tqdm(trainLoader) as iterator:
        iterator.set_description("Epoch {}".format(i))
        for idx,data in enumerate(iterator):
            input, projection, label = data
            input, projection, label = input.to(device), projection.to(device), label.to(device)
            output = net(input, projection)
            if idx % 100 == 0:
                draw(output, debugPath, label)
            optimizer.zero_grad()
            loss =lossFunction(output[-1], label)
            loss.backward()
            optimizer.step()
            trainLoss.append(loss.item())
            iterator.set_postfix_str(
                "loss:{},epoch mean:{:.2f}".format(loss.item(), np.mean(np.array(trainLoss))))
    scheduler.step()
    if i%10==0: torch.save({
        "epoch": i, "model": net.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()
    }, "{}/dtvnet_v2.5cq_{:.10f}.dict".format(checkpointPath, np.mean(np.array(trainLoss))))
