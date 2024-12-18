import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
