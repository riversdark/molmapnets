# AUTOGENERATED! DO NOT EDIT! File to edit: 01_nets.ipynb (unless otherwise specified).

__all__ = ['Convnet', 'Inception', 'DoubleInception', 'SinglePathFullyConnected', 'DoublePathFullyConnected',
           'SinglePathMolMapNet', 'DoublePathMolMapNet', 'Resnet']

# Cell
import torch
from torch import nn
import torch.nn.functional as F

# Cell
class Convnet(nn.Module):
    "Convolutional feature extraction Block"
    def __init__(self, C_in=13, C_out=48, conv_size=13):
        super(Convnet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=conv_size, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):

        return self.conv(x)

# Cell
class Inception(nn.Module):
    "Naive Google Inception Block"
    def __init__(self, C_in=48, C_out=32, stride=1):
        super(Inception, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=5, stride=stride, padding='same'),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=3, stride=stride, padding='same'),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, padding='same'),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        return torch.cat((x1, x2, x3), dim=1)

# Cell
class DoubleInception(nn.Module):
    "Double Inception Block"
    def __init__(self, C_in1=48, C_out1=32, stride1=1, C_in2=96, C_out2=64, stride2=1):
        super(DoubleInception, self).__init__()

        self.inception1 = Inception(C_in1, C_out1, stride1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception2 = Inception(C_in2, C_out2, stride2)

    def forward(self, x):
        x = self.inception1(x)
        x = self.maxpool(x)
        x = self.inception2(x)

        return x

# Cell
class SinglePathFullyConnected(nn.Module):
    "Fully connected layers for single path MolMap nets"
    def __init__(self, C1=192, C2=128, C3=32):
        super(SinglePathFullyConnected, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(C1, C2),
            nn.ReLU(),
            nn.Linear(C2, C3)
        )

    def forward(self, x):
        return self.fc(x)

# Cell
class DoublePathFullyConnected(nn.Module):
    "Fully connected layers for double paths MolMap nets"
    def __init__(self, C1=384, C2=256, C3=128, C4=32):
        super(DoublePathFullyConnected, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(C1, C2),
            nn.ReLU(),
            nn.Linear(C2, C3),
            nn.ReLU(),
            nn.Linear(C3, C4),
        )

    def forward(self, x):
        return self.fc(x)

# Cell
class SinglePathMolMapNet(nn.Module):
    "Single Path Molecular Mapping Network"
    def __init__(self, conv_in=13, conv_size=13, FC=[128, 32]):
        super(SinglePathMolMapNet, self).__init__()

        # output channels in the double inception
        C_out1, C_out2 = 32, 64

        self.conv = Convnet(C_in=conv_in, C_out=48, conv_size=conv_size)
        self.double_inception = DoubleInception(C_in1=48, C_out1=C_out1, C_in2=C_out1*3, C_out2=C_out2)
        self.fully_connected = SinglePathFullyConnected(C1=C_out2*3, C2=FC[0], C3=FC[1])

    def forward(self, x):
        x = self.conv(x)
        x = self.double_inception(x)
        x = x.amax(dim=(-1, -2))
        x = self.fully_connected(x)

        return x

# Cell
class DoublePathMolMapNet(nn.Module):
    "Double Path Molecular Mapping Network"
    def __init__(self, conv_in1=13, conv_in2=3, conv_size=13, FC=[256, 128, 32]):
        super(DoublePathMolMapNet, self).__init__()

        # output channels in the double inception
        C_out1, C_out2 = 32, 64

        self.conv1 = Convnet(C_in=conv_in1, C_out=48, conv_size=conv_size)
        self.conv2 = Convnet(C_in=conv_in2, C_out=48, conv_size=conv_size)
        self.double_inception = DoubleInception(C_in1=48, C_out1=C_out1, C_in2=C_out1*3, C_out2=C_out2)
        self.fully_connected = DoublePathFullyConnected(C1=C_out2*6, C2=FC[0], C3=FC[1], C4=FC[2])

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.double_inception(x1)
        x1 = x1.amax(dim=(-1, -2))

        x2 = self.conv2(x2)
        x2 = self.double_inception(x2)
        x2 = x2.amax(dim=(-1, -2))

        x = torch.cat((x1, x2), dim=1)
        x = self.fully_connected(x)

        return x

# Cell
class Resnet(nn.Module):
    "Naive Google Inception Block"
    def __init__(self, C, conv_size):
        super(Resnet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=conv_size, stride=1, padding='same'),
            nn.BatchNorm2d(C),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=conv_size, stride=1, padding='same'),
            nn.BatchNorm2d(C)
        )

    def forward(self, x):
        o = self.conv1(x)
        o = self.conv2(o)
        o += x

        return F.relu(o)