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
import sys
import cv2
import pdb
from PIL import Image
import matplotlib.pyplot as plt 

from dataloader import MyDataset
from model import MobileNetv2_SISR, Mobile_UNet
from utils import *
from train import *

'''
Sample commands -> python3 main.py 100 32 -d_set M -channels 3 --train /--test
               -> python3 main.py 51 32 -d_set C -channels 3 --train --transfer >> tl_cm
TODO: [*] Arg Parse for test pred
TODO: [*] Try U-Net
TODO: [*] Performance with different d_set sizes eg- medium_10k
            - 1k, 10k, 20k, 50k
TODO: [*] Validation and train plots
TODO: [*] Intermediate activations save on test data
TODO: [*] Mobile Unet
         - Pixel Shuffle Upsampling
         - Unet - Encoder and Decoder
         - BottleNeck Inverted Residual Layers w/ GSC (4x2)
 '''        

if __name__ == '__main__':

    root = "data/"
    d_set = {'F': 'k11', 'M': 'k21', 'C': 'k41'}
    mode_dict = {'1': '', '3': '_3c'}

    parser = argparse.ArgumentParser()
    parser.add_argument('-d_set', dest='dataset', default='M', help="Select which dataset to use: fine (F), medium(M) or coarse(C)")
    parser.add_argument('-channels', dest='mode', default='3', help="Select Mode for model prediction: 1c, 3c")
    parser.add_argument(dest='epoch', type=int, default=100, help= 'Number of Epochs')
    parser.add_argument(dest='batch_size', type=int, default=32, help= 'Batch Size')
    parser.add_argument('--test', dest='testing', action='store_true')
    parser.add_argument('--train', dest='training', action='store_true')
    parser.add_argument('--transfer', dest='transfer_learning', action='store_true')

    args = parser.parse_args()
    root += d_set.get(args.dataset)
    mode = mode_dict.get(args.mode)

    # print(root, mode)
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
    train = 4e4
    dev = 1e3
    test = 1e3

    data_split = (train, dev, test)

    np.random.seed(42)

    img_list = np.random.permutation(int(sum(data_split)))
    # img_list = range(0, int(sum(data_split)+1), 1)
    # print(img_list)
    # sys.exit()

    model = Mobile_UNet()

    if args.training:

        train_dataset = MyDataset('train', root, img_list, data_split, transform=img_transform)
        train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=8)
        train_loader = data.DataLoader(train_dataset, **train_loader_args)
        # print(train_dataset.__len__())

        dev_dataset = MyDataset('dev', root, img_list, data_split, transform=test_transform)
        dev_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=8)
        dev_loader = data.DataLoader(dev_dataset, **dev_loader_args)
        # print(dev_dataset.__len__())

        test_dataset = MyDataset('test', root, img_list, data_split, transform=test_transform)
        test_loader_args = dict(batch_size=1, shuffle=False, num_workers=8)
        test_loader = data.DataLoader(test_dataset, **test_loader_args)
        # print(test_dataset.__len__())

        if args.transfer_learning:

            tl_root = 'data/k21'
            tl_epoch = 45
            PATH = f'{tl_root}/model_3c/SISR_mv2f_{tl_epoch}.pth'
            model.load_state_dict(torch.load(PATH))
            device = torch.device("cuda")
            model.eval()
            model.to(device)
            print(f'Using Transfer Learning')

        else:

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

            scheduler.step(train_loss)
            print(optimizer)
            print('='*100)

            if epoch%5 == 0:
                torch.save(model.state_dict(), f'{root}/model_3c/SISR_mv2f_{epoch}.pth')

        plot_training(Train_Loss, Dev_Loss, root)
        np.save('train_loss.npy', np.array(Train_Loss))
        np.save('dev_loss.npy', np.array(Dev_Loss))

        print(f'Prediction at Epoch: {num_epochs}')
        test_predictions(model, test_loader, root)
        # train_predictions(model, test_loader)
            

    if args.testing:

        test_dataset = MyDataset('test', root, img_list, data_split, transform=test_transform)
        test_loader_args = dict(batch_size=1, shuffle=False, num_workers=8)
        test_loader = data.DataLoader(test_dataset, **test_loader_args)
        # print(test_dataset.__len__())

        test_epoch = 30
        PATH = f'{root}/model_3c/SISR_mv2f_{test_epoch}.pth'
        model.load_state_dict(torch.load(PATH))
        device = torch.device("cuda")
        model.eval()
        model.to(device)

        print(f'Prediction at Epoch: {num_epochs}')
        t1 = time.time()
        test_predictions(model, test_loader, root)
        # train_predictions(model, test_loader)

