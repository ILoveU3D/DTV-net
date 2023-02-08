import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.trainDataLoader import Stimulated256
from model.PDHG.LEARN import LEARN
from options import trainPath, validPath
from loss import draw
from loss import stepLoss as lossFunction

trainSet = Stimulated256(trainPath)
validSet = Stimulated256(validPath)
trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True)
validLoader = DataLoader(validSet, batch_size=1, shuffle=False)

device = 4
debugPath = "/home/nanovision/wyk/data/debug_learn/"
checkpointPath = "/home/nanovision/wyk/data/checkpoints_learn/"
net = LEARN(30).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
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
            if idx % 100 == 0: draw(output[-9:-1], debugPath)
            optimizer.zero_grad()
            loss = lossFunction(output, label.to(device))
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
                loss = lossFunction(output, label.to(device))
                validLoss.append(loss.item())
                iterator.set_postfix_str(
                    "loss:{},epoch mean:{:.2f}".format(loss.item(), np.mean(np.array(validLoss))))
    scheduler.step()
    if i%10==0: torch.save({
        "epoch": i, "model": net.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()
    }, "{}/learn_{:.10f}.dict".format(checkpointPath, np.mean(np.array(trainLoss))))