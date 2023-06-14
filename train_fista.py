import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.simulateLoader import Stimulated256Input
from data.realLoader import Real256Input
from model.FISTA.FISTAnet import FistaNet
from options import trainPath, validPath, inputTrainData, inputValidData
from loss import draw
# from loss import stepLoss as lossFunction
lossFunction = torch.nn.L1Loss(reduction =  "mean")

device = 4
#trainSet = Stimulated256Input(trainPath, inputTrainData, device)
#validSet = Stimulated256Input(validPath, inputValidData, device)
#trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True)
#validLoader = DataLoader(validSet, batch_size=1, shuffle=False)
trainPath = ["/home/nanovision/wyk/data/real/1", "/home/nanovision/wyk/data/real/2", "/home/nanovision/wyk/data/real/3"]
trainSet = Real256Input(trainPath, device)
trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True)

debugPath = "/home/nanovision/wyk/data/debug_fista/"
checkpointPath = "/home/nanovision/wyk/data/checkpoints_fista/"
pretrain = "/home/nanovision/wyk/data/checkpoints_fista/fista_net_64485.3274356618.dict"
net = FistaNet(5).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
# dictionary = torch.load(pretrain)
# net.load_state_dict(dictionary["model"])
# optimizer.load_state_dict(dictionary["optimizer"])
# scheduler.load_state_dict(dictionary["scheduler"])

epoch = 50

for i in range(epoch):
    trainLoss = []
    validLoss = []
    net.train()
    with tqdm(trainLoader) as iterator:
        iterator.set_description("Epoch {}".format(i))
        for idx,data in enumerate(iterator):
            input, projection, label = data
            output = net(input.to(device), projection.to(device))
            label = label.to(device)
            if idx % 100 == 0: draw(output, debugPath, label)
            optimizer.zero_grad()
            loss = lossFunction(output[-1], label)
            loss.backward()
            optimizer.step()
            trainLoss.append(loss.item())
            iterator.set_postfix_str(
                "loss:{},epoch mean:{:.2f}".format(loss.item(), np.mean(np.array(trainLoss))))
    scheduler.step()

torch.save({"epoch": i, "model": net.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, "{}/fista_net_real{:.10f}.dict".format(checkpointPath, np.mean(np.array(trainLoss))))
