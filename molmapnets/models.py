# AUTOGENERATED! DO NOT EDIT! File to edit: 02_models.ipynb (unless otherwise specified).

__all__ = ['MolMapRegression', 'MolMapMultiClassClassification', 'MolMapMultiLabelClassification']

# Cell
import torch
from torch import nn
import torch.nn.functional as F

from .nets import SinglePathMolMapNet, DoublePathMolMapNet

# Cell
class MolMapRegression(nn.Module):
    "Mol Map nets used for regression"
    def __init__(self, conv_in1=13, conv_in2=None, conv_size=13):
        super(MolMapRegression, self).__init__()

        if conv_in2 is None:
            self.net = SinglePathMolMapNet(conv_in=conv_in1, FC=[128, 32])
            self.single = True
        else:
            self.net = DoublePathMolMapNet(conv_in1=conv_in1, conv_in2=conv_in2, FC=[256, 128, 32])
            self.single = False

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        "x: Tensor or tuple of Tensors"
        if self.single:
            x = self.net(x)
        else:
            x1, x2 = x
            x = self.net(x1, x2)

        return self.fc(x)

# Cell
class MolMapMultiClassClassification(nn.Module):
    "MolMap nets used for multi-class classification"
    def __init__(self, conv_in1=13, conv_in2=None, conv_size=13, n_class=10):
        super(MolMapMultiClassClassification, self).__init__()

        if conv_in2 is None:
            self.net = SinglePathMolMapNet(conv_in=conv_in1, FC=[128, 32])
            self.single = True
        else:
            self.net = DoublePathMolMapNet(conv_in1=conv_in1, conv_in2=conv_in2, FC=[256, 128, 32])
            self.single = False

        self.fc = nn.Linear(32, n_class)

    def forward(self, x):
        "x: Tensor or tuple of Tensors"
        if self.single:
            x = self.net(x)
        else:
            x1, x2 = x
            x = self.net(x1, x2)

        x = self.fc(x)

        return F.log_softmax(x, dim=1)

# Cell
class MolMapMultiLabelClassification(nn.Module):
    "MolMap nets used for multi-label classification"
    def __init__(self, conv_in1=13, conv_in2=None, conv_size=13, n_label=5):
        super(MolMapMultiLabelClassification, self).__init__()

        if conv_in2 is None:
            self.net = SinglePathMolMapNet(conv_in=conv_in1, FC=[128, 32])
            self.single = True
        else:
            self.net = DoublePathMolMapNet(conv_in1=conv_in1, conv_in2=conv_in2, FC=[256, 128, 32])
            self.single = False

        self.fc = nn.Linear(32, n_label)

    def forward(self, x):
        "x: Tensor or tuple of Tensors"
        if self.single:
            x = self.net(x)
        else:
            x1, x2 = x
            x = self.net(x1, x2)

        x = self.fc(x)

        return torch.sigmoid(x)