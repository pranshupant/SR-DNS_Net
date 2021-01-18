import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim import lr_scheduler
import time
import os
import cv2
import pdb
from PIL import Image
import matplotlib.pyplot as plt 

from dataloader import MyDataset, LES_DATA_PATH, DNS_DATA_PATH, ROOT
from model import MobileNetv2_SISR
from utils import *
from train import *

if not os.path.exists('model_3c'):
    os.mkdir('model_3c')

if not os.path.exists('test_3c'):
    os.mkdir('test_3c')

num_epochs = 100
batch_size = 32
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.05),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = MyDataset('train', transform=img_transform)
train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=8)
train_loader = data.DataLoader(train_dataset, **train_loader_args)
# print(train_dataset.__len__())

dev_dataset = MyDataset('dev', transform=test_transform)
dev_loader_args = dict(batch_size=batch_size, shuffle=False, num_workers=8)
dev_loader = data.DataLoader(dev_dataset, **dev_loader_args)
# print(dev_dataset.__len__())

test_dataset = MyDataset('test', transform=test_transform)
test_loader_args = dict(batch_size=1, shuffle=False, num_workers=8)
test_loader = data.DataLoader(test_dataset, **test_loader_args)
# print(test_dataset.__len__())

model = MobileNetv2_SISR()
model.apply(MobileNetv2_SISR.init_weights)
device = torch.device("cuda")
model.eval()
model.to(device)        
# print(model)

Train_Loss = []
Dev_Loss = []
Dev_Acc = []
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience=3, threshold=5e-4, eps=1e-6)

for epoch in range(num_epochs):

    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    dev_loss, psnr_les, psnr, ssim_les, ssim_dns = dev_epoch(model, dev_loader, criterion)
    Train_Loss.append(train_loss)
    Dev_Loss.append(dev_loss)

    scheduler.step(dev_loss)

    print(' ')
    print('epoch [{}/{}], Train_Loss:{:.6f}, Dev_Loss:{:.6f}'.format(epoch+1, num_epochs, train_loss, dev_loss))
    print('PSNR_DNS:{:.4f}, PSNR_LES:{:.4f}, SSIM_LES:{:.4f}, SSIM_DNS:{:.4f}'.format(psnr, psnr_les, ssim_les, ssim_dns))

    torch.save(model.state_dict(), 'model_3c/SISR_mv2f_{}.pth'.format(epoch))
    scheduler.step(train_loss)
    print(optimizer)
    print('='*100)

batch_size = 1
avg_psnr, avg_psnr_les, avg_les_ke, avg_recon_ke, avg_dns_ke, avg_ssim_les, avg_ssim_dns = test_predictions(model, test_loader, batch_size)
print(avg_psnr)
print(avg_psnr_les)
print(avg_recon_ke)
print(avg_dns_ke)
print(avg_les_ke)
print(avg_ssim_les)
print(avg_ssim_dns)

