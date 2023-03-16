import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.simulateLoader import Stimulated256InputShepp
from model.FISTA.DTVSTAnet import DTVNet
from model.FISTA.RegularizationLayers.RED import Red
from options import trainPath, inputTrainData, validPath, inputValidData, checkpointPath, debugPath, pretrain3
from loss import draw
from loss import stepLoss as lossFunction
debugPath = "/home/nanovision/wyk/data/train4"

trainPath = "/home/nanovision/wyk/data/shepp"
inputTrainData = "/home/nanovision/wyk/data/sheppInput"
validPath = "/home/nanovision/wyk/data/shepp"
inputValidData = "/home/nanovision/wyk/data/sheppInput"
trainSet = Stimulated256InputShepp(trainPath, inputTrainData)
validSet = Stimulated256InputShepp(validPath, inputValidData)
trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True)
validLoader = DataLoader(validSet, batch_size=1, shuffle=False)

device = 3
net = DTVNet((256,256,64),5).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
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
            loss =lossFunction(output, label)
            loss.backward()
            optimizer.step()
            trainLoss.append(loss.item())
            iterator.set_postfix_str(
                "loss:{},epoch mean:{:.2f}".format(loss.item(), np.mean(np.array(trainLoss))))
    net.eval()
    with torch.no_grad():
        with tqdm(validLoader) as iterator:
            iterator.set_description("validating...")
            for idx, data in enumerate(iterator):
                input, projection, label = data
                input, projection, label = input.to(device), projection.to(device), label.to(device)
                output = net(input, projection)
                loss = lossFunction(output, label)
                validLoss.append(loss.item())
                iterator.set_postfix_str(
                    "loss:{},epoch mean:{:.2f}".format(loss.item(), np.mean(np.array(validLoss))))
    scheduler.step()
    if i%10==0: torch.save({
        "epoch": i, "model": net.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()
    }, "{}/dtvnet_v2.2shp_{:.10f}.dict".format(checkpointPath, np.mean(np.array(trainLoss))))
