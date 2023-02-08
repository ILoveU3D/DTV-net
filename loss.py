import torch
import torch.nn as nn
from PIL import Image
import torchvision

def KLDivLoss(output, predict, mean, var):
    mse = nn.MSELoss(reduction='mean')(output, predict)
    kld = -0.5 * torch.mean((1+var-mean.pow(2)-torch.exp(var)).view(-1))
    return mse + kld, mse.item(), kld.item()

def stepLoss(outputs, label):
    loss = 0
    weight = 1
    for output in outputs:
        loss += weight * nn.MSELoss(reduction="mean")(output, label)
        weight *= 0.9
    return loss

def draw(output, path, label):
    target = Image.new('L', (256*3, 256*3))
    for row in range(3):
        for col in range(3):
            if row*3+col < len(output):
                img = output[row*3+col][0,0,32,:,:].detach().cpu().numpy()
                img = (img - img.ravel().min())/(img.ravel().max() - img.ravel().min() + 1)
                img = Image.fromarray(img * 255)
                target.paste(img, (col*256,row*256))
    target.save("{}/output.jpg".format(path))
    output[-1].detach().cpu().numpy().tofile("{}/output.raw".format(path))
    label.detach().cpu().numpy().tofile("{}/label.raw".format(path))

def perceptualLossCal(image, label):
    perceptualSum = 0
    vgg19 = torchvision.models.vgg19(weights="DEFAULT").features.to(image.device)
    for i in range(image.size(2)):
        img = image[...,i,:,:].expand(-1,3,-1,-1)
        lab = label[...,i,:,:].expand(-1,3,-1,-1)
        perceptualSum += nn.L1Loss(reduction="mean")(vgg19(img), vgg19(lab))
    return nn.MSELoss(reduction="mean")(image, label) + 10e-3 * perceptualSum