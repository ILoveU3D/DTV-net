import numpy as np
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """docstring for  BasicBlock"""

    def __init__(self, features=32):
        super(BasicBlock, self).__init__()
        self.Sp = nn.Softplus()

        self.conv_D = nn.Conv3d(1, features, (3, 3, 3), stride=1, padding=1)
        self.conv1_forward = nn.Conv3d(features, features, (3, 3, 3), stride=(1, 2, 2), padding=1)
        self.conv2_forward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)
        self.conv3_forward = nn.Conv3d(features, features, (3, 3, 3), stride=(1, 2, 2), padding=1)
        self.conv4_forward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)

        self.conv1_backward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)
        self.conv2_backward = nn.Conv3d(features*2, features, (3, 3, 3), stride=1, padding=1)
        self.conv3_backward = nn.Conv3d(features*2, features, (3, 3, 3), stride=1, padding=1)
        self.conv4_backward = nn.Conv3d(features*2, features, (3, 3, 3), stride=1, padding=1)
        self.conv_G = nn.Conv3d(features*2, 1, (3, 3, 3), stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')

    def forward(self, x):
        x_input = x  #256
        x_D = self.conv_D(x_input) #256

        l1 = self.conv1_forward(x_D) #128
        l1 = self.relu(l1)
        l2 = self.conv2_forward(l1) #128
        l2 = self.relu(l2)
        l3 = self.conv3_forward(l2) #64
        l3 = self.relu(l3)
        x_forward = self.conv4_forward(l3) #64
        
        l1b = self.conv1_backward(x_forward) #64
        l1b = self.relu(l1b)

        l1b = torch.cat([l3, l1b], 1)
        l2b = self.conv2_backward(l1b) #64
        l2b = self.relu(l2b)

        l2b = self.upsample(l2b)      #128
        l2b = torch.cat([l2, l2b], 1)
        l3b = self.conv3_backward(l2b)
        l3b = self.relu(l3b)

        l3b = torch.cat([l1, l3b], 1) #128
        x_backward = self.conv4_backward(l3b)
        x_backward = self.relu(x_backward)

        x_backward = self.upsample(x_backward)
        x_backward = torch.cat([x_D, x_backward], 1) #256
        x_G = self.conv_G(x_backward)

        # prediction output (skip connection); non-negative output
        x_pred = self.relu(x_input + x_G)

        return x_pred

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_normal_(m.weight)
