import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim import lr_scheduler
import argparse
import time
import os
import cv2
import pdb
from PIL import Image
import matplotlib.pyplot as plt 

from dataloader import MyDataset
from model import MobileNetv2_SISR
from utils import *
from train import *

# Sample command -> python3 main.py 100 32 -d F -m 3

if __name__ == '__main__':

    root = "data/"
    d_set = {'F': 'k11', 'M': 'k21', 'C': 'k41'}
    mode_dict = {'1': '', '3': '_3c'}

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset', default='M', help="Select which dataset to use: fine (F), medium(M) or coarse(C)")
    parser.add_argument('-m', dest='mode', default='3', help="Select Mode for model prediction: 1c, 3c")
    parser.add_argument(dest='epoch', type=int, default=100, help= 'Number of Epochs')
    parser.add_argument(dest='batch_size', type=int, default=32, help= 'Batch Size')


    args = parser.parse_args()
    root += d_set.get(args.dataset)
    mode = mode_dict.get(args.mode)

    print(root, mode)
    print(args.epoch, args.batch_size)

    create_directories(root, mode)

    num_epochs = args.epoch
    batch_size = args.batch_size
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

    #train, dev, test stplit
    train = 2e4
    dev = 1e3
    test = 1e3

    data_split = (train, dev, test)

    img_list = np.random.permutation(int(sum(data_split)))

    train_dataset = MyDataset('train', root, img_list, data_split, transform=img_transform)
    train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=8)
    train_loader = data.DataLoader(train_dataset, **train_loader_args)
    # print(train_dataset.__len__())

    dev_dataset = MyDataset('dev', root, img_list, data_split, transform=test_transform)
    dev_loader_args = dict(batch_size=batch_size, shuffle=False, num_workers=8)
    dev_loader = data.DataLoader(dev_dataset, **dev_loader_args)
    # print(dev_dataset.__len__())

    test_dataset = MyDataset('test', root, img_list, data_split, transform=test_transform)
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
 
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience=3, threshold=5e-4, eps=1e-6)

    for epoch in range(num_epochs):

        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        dev_loss = dev_epoch(model, dev_loader, criterion)
        Train_Loss.append(train_loss)
        Dev_Loss.append(dev_loss)

        scheduler.step(dev_loss)

        print(' ')
        print(f'epoch [{epoch+1}/{num_epochs}], Train_Loss:{train_loss:.6f}, Dev_Loss:{dev_loss:.6f}')
        
        torch.save(model.state_dict(), f'{root}/model_3c/SISR_mv2f_{epoch}.pth')
        scheduler.step(train_loss)
        print(optimizer)
        print('='*100)

    test_predictions(model, test_loader, root)
    # train_predictions(model, test_loader)


