import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.simulateLoader import Stimulated256Input
from model.FISTA.FISTAnet import FistaNet
from options import trainPath, validPath, inputTrainData, inputValidData
from loss import draw
# from loss import stepLoss as lossFunction
lossFunction = torch.nn.L1Loss(reduction =  "mean")

trainSet = Stimulated256Input(trainPath, inputTrainData)
validSet = Stimulated256Input(validPath, inputValidData)
trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True)
validLoader = DataLoader(validSet, batch_size=1, shuffle=False)

device = 5
debugPath = "/home/nanovision/wyk/data/debug_fista/"
checkpointPath = "/home/nanovision/wyk/data/checkpoints_fista/"
pretrain = "/home/nanovision/wyk/data/checkpoints_fista/fista_net_64485.3274356618.dict"
net = FistaNet(5).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
# dictionary = torch.load(pretrain)
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
            output = net(input.to(device), projection.to(device))
            if idx % 100 == 0: draw(output, debugPath, label)
            optimizer.zero_grad()
            loss = lossFunction(output[-1], label.to(device))
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
                output = net(input.to(device), projection.to(device))
                loss = lossFunction(output[-1], label.to(device))
                validLoss.append(loss.item())
                iterator.set_postfix_str(
                    "loss:{},epoch mean:{:.2f}".format(loss.item(), np.mean(np.array(validLoss))))
    scheduler.step()
    if i%10==0: torch.save({
        "epoch": i, "model": net.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()
    }, "{}/fista_net_{:.10f}.dict".format(checkpointPath, np.mean(np.array(trainLoss))))
