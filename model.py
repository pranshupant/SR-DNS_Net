import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import os
import pdb

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetv2_SISR(nn.Module):
    def __init__(self):
        super(MobileNetv2_SISR, self).__init__()

        self.conv1 = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),  # b, 32, 64, 64
        nn.BatchNorm2d(32),
        nn.LeakyReLU()
        )
        self.bottlenecks1 = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 16, 1, 1),
            InvertedResidual(16, 24, 1, 6),
            InvertedResidual(24, 24, 1, 6),
            InvertedResidual(24, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
        )

        self.bottlenecks2 = nn.Sequential(
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 96, 2, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 128, 1, 6),
            InvertedResidual(128, 128, 1, 6),
            InvertedResidual(128, 128, 1, 6),
            InvertedResidual(128, 256, 1, 6),
            #InvertedResidual(256, 256, 1, 6),
        )
        self.deconv1 = nn.Sequential(
        nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),  # b, 1, 64, 64
        #nn.Tanh()
        )
        self.pix_shuffle = nn.PixelShuffle(2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv1(x)
        
        out = self.bottlenecks1(out)
        int1 = out
        out = self.bottlenecks2(out)
        out = self.pix_shuffle(out)
        out = torch.add(out, int1)

        out = self.deconv1(out)
        out1 = self.tanh(out)+0.0001*out
        return out1

        #return out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)