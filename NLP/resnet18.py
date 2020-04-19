'''ResNet-18 Image classfication for cifar-10 with PyTorch 

Author 'Sun-qian'.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(3072, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        print("strides: ", strides)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        print("x shape: ", x.shape)
        out = self.conv1(x)
        print("out1 shape: ", out.shape)
        out = self.layer1(out)
        print("out2 shape: ", out.shape)
        out = self.layer2(out)
        print("out3 shape: ", out.shape)
        out = self.layer3(out)
        print("out4 shape: ", out.shape)
        out = self.layer4(out)
        print("out5 shape: ", out.shape)
        out = F.avg_pool2d(out, 4)
        print("out6 shape: ", out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        print("out7 shape: ", out.shape)
        return out


def ResNet18():
    return ResNet(ResidualBlock)

if __name__ == "__main__":
    res = ResNet18()
    a = torch.randn(1, 3, 100, 64)
    b = res(a)
    print(b.shape)