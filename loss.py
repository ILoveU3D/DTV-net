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
        loss += weight * nn.L1Loss(reduction="mean")(output, label)
        weight *= 1.2
    return loss

def _getSlice(img, p=1, sliceNum=32):
    img = img[0,0,sliceNum,:,:].detach().cpu().numpy()
    img = p * (img - img.ravel().min())/(img.ravel().max() - img.ravel().min() + 1)
    img = Image.fromarray(img * 255)
    return img

def draw(output, path, label):
    target = Image.new('L', (256*3, 256*4))
    for row in range(3):
        for col in range(3):
            if row*3+col < len(output):
                img = _getSlice(output[row*3+col])
                target.paste(img, (col*256,row*256))
    labelImg = _getSlice(label)
    target.paste(labelImg, (0,3*256))
    diff = _getSlice(torch.abs(label-output[-1]))
    target.paste(diff, (256,3*256))
    diff10 = _getSlice(torch.abs(label-output[-1]), p=10)
    target.paste(diff10, (2*256,3*256))
    target.save("{}/output.jpg".format(path))
    output[-1].detach().cpu().numpy().tofile("{}/output.raw".format(path))
    label.detach().cpu().numpy().tofile("{}/label.raw".format(path))

def drawLatent(output, path):
    for i,l in enumerate(output):
        target = Image.new('L', (256*4, 256*8))
        mm = l[0,17,32,:,:].detach().cpu().numpy().ravel().max()
        mi = l[0,17,32,:,:].detach().cpu().numpy().ravel().min()
        for row in range(4):
            for col in range(8):
                img = l[0,row*4+col,32,:,:].detach().cpu().numpy()
                img = (img-mi) / (mm-mi)
                img = Image.fromarray(img * 255).resize((256,256), Image.ANTIALIAS)
                target.paste(img, (row*256, col*256))
        target.save("{}/output_{}.jpg".format(path, i))

def perceptualLossCal(image, label):
    perceptualSum = 0
    vgg19 = torchvision.models.vgg19(weights="DEFAULT").features.to(image.device)
    for i in range(image.size(2)):
        img = image[...,i,:,:].expand(-1,3,-1,-1)
        lab = label[...,i,:,:].expand(-1,3,-1,-1)
        perceptualSum += nn.L1Loss(reduction="mean")(vgg19(img), vgg19(lab))
    return nn.MSELoss(reduction="mean")(image, label) + 10e-3 * perceptualSum
