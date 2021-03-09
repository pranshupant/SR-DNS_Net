import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import os
import pdb
from utils import save_activation

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

############# U-Net #############   

class conv_bn_LRelu(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding): 
        super(conv_bn_LRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel, stride, padding, groups=C_in),
            nn.Conv2d(C_in, C_out, 1, 1, 0),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x

class convTranspose_bn_LRelu(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding): 
        super(convTranspose_bn_LRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_in, kernel, stride, padding, groups=C_in),
            nn.ConvTranspose2d(C_in, C_out, 1, 1, 0),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x 
    
class Downsample(nn.Module):
    def __init__(self, C_in, C_out, kernel=4, stride=2, padding=1): 
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel, stride, padding, groups=C_in),
            nn.Conv2d(C_in, C_out, 1, 1, 0),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, C_in, C_out, kernel=4, stride=2, padding=1): 
        super(Upsample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_in, kernel, stride, padding, groups=C_in),
            nn.ConvTranspose2d(C_in, C_out, 1, 1, 0)
        )
        self.bn=nn.BatchNorm2d(C_out)
        self.lRelu = nn.LeakyReLU()

    def forward(self,x1,x2=None,last=False):
        if x2 is None:
            x=self.layer(x1)
        else:
            x=torch.cat((x1,x2),dim=1)
            x=self.layer(x)
        
        if last==False:
            x=self.lRelu(self.bn(x))
        return x


class Upsample_PS(nn.Module):
    def __init__(self, C_in, C_out): 
        super(Upsample_PS, self).__init__()
        self.layer = nn.Sequential(
            nn.PixelShuffle(2),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU()
        )

    def forward(self,x1,x2):

        x=torch.cat((x1,x2),dim=1)
        x=self.layer(x)

        return x

class Mobile_UNet(nn.Module):
    def __init__(self):
        super(Mobile_UNet, self).__init__()

        #encoder
        self.d1=Downsample(3,32)    #16, 64, 64
        self.d2=Downsample(32,64)   #32, 32, 32
        self.d3=Downsample(64,128)   #128, 16, 16
        # self.d4=Downsample(64,128)
        # self.d5=Downsample(128,256)

        self.bottleneck1 = nn.Sequential(
            # nn.Identity()
            InvertedResidual(128, 128, 1, 6), #128, 16, 16
            InvertedResidual(128, 128, 1, 6), #128, 16, 16
            InvertedResidual(128, 128, 1, 6), #128, 16, 16
            InvertedResidual(128, 128, 1, 6), #128, 16, 16
        #     InvertedResidual(128, 128, 1, 6), #128, 16, 16
        #     InvertedResidual(128, 128, 1, 6), #128, 16, 16
        #     InvertedResidual(128, 128, 1, 6), #128, 16, 16
        #     InvertedResidual(128, 128, 1, 6), #128, 16, 16
        )

        self.bottleneck2 = nn.Sequential(
            # nn.Identity()
            InvertedResidual(128, 128, 1, 6), #128, 16, 16
            InvertedResidual(128, 128, 1, 6), #128, 16, 16
            InvertedResidual(128, 128, 1, 6), #128, 16, 16
            InvertedResidual(128, 128, 1, 6), #128, 16, 16
            # InvertedResidual(128, 128, 1, 6), #128, 16, 16
            # InvertedResidual(128, 128, 1, 6), #128, 16, 16
            # InvertedResidual(128, 128, 1, 6), #128, 16, 16
            # InvertedResidual(128, 128, 1, 6), #128, 16, 16
        )

        self.u1=Upsample_PS(128,64) #128, 32, 32
        self.u2=Upsample_PS(64,32) #64, 64, 64
        self.u3=Upsample_PS(32,16) #32, 128, 128
        # self.u4=Upsample(64,16)
        # self.u5=Upsample(32,1,(3,8),(1,8),(1,0))
        self.deconv = nn.ConvTranspose2d(16, 3, 1, 1, 0)
        self.tanh = nn.Tanh()
        
    def forward(self,x):

        save_activation(x, 0, 1)
        down1=self.d1(x)
        save_activation(down1, 1, 2)
        down2=self.d2(down1)
        save_activation(down2, 2, 4)
        down3=self.d3(down2)
        save_activation(down3, 3, 8)
        # down4=self.d4(down3)
        # down5=self.d5(down4)
        bn1=self.bottleneck1(down3)
        save_activation(bn1, 4, 8)
        mid1 = down3 + bn1
        bn2=self.bottleneck2(mid1)
        save_activation(bn2, 5, 8)
        mid2 = mid1 + bn2

        up1=self.u1(down3,mid2)
        save_activation(up1, 6, 4)
        # print(up1.shape)
        up2=self.u2(down2,up1)
        save_activation(up2, 7, 2)
        up3=self.u3(down1,up2)
        save_activation(up3, 8, 1)
        # up4=self.u4(down2,up3)
        # up5=self.u5(down1,up4,last=True)
        out = self.deconv(up3)
        out = self.tanh(out)
        save_activation(out, 9, 1)

        return out