import numpy as np
import math
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image
import time
import os
import cv2
import pdb
from PIL import Image
import matplotlib.pyplot as plt
from utils import *

batch_size = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## TODO : [] Remove batch_size decl.
##      : [] Print within the fucntions not main

def train_epoch(model, data_loader, criterion, optimizer):
    model.train()
    
    running_loss = 0
    avg_psnr = 0
    avg_psnr_les = 0

    start_time = time.time()
    print('Train Loop')
    for batch_num, (img, target) in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True):

        img = img.to(device)
        target = target.to(device)

        output = model(img)
        loss = criterion(output, target)
        psnr = PSNR(output, target, batch_size)
        psnr_les = PSNR(img, target, batch_size)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()/len(data_loader)
        avg_psnr += psnr.item()/len(data_loader)
        avg_psnr_les += psnr_les.item()/len(data_loader)

    #print(' ')
    print('Train_Loss:{:.6f}, PSNR_DNS:{:.4f}, PSNR_LES:{:.4f}'.format(running_loss, psnr, psnr_les))
    torch.cuda.empty_cache()
    end_time = time.time()
    del img
    del target
    del loss
    print("Train Time: {:.2f} s".format(end_time-start_time))

    return running_loss


def dev_epoch(model, data_loader, criterion):
    with torch.no_grad():
        model.eval()

        running_loss = 0
        avg_psnr = 0
        avg_psnr_les = 0
        avg_ssim_dns = 0
        avg_ssim_les = 0

        start_time = time.time()
        print('Dev Loop')
        print(' ')

        for batch_num, (img, target) in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True):

            img = img.to(device)
            target = target.to(device)

            output = model(img)
            loss = criterion(output, target)
            psnr = PSNR(output, target,batch_size)
            psnr_les = PSNR(img, target,batch_size)
            ssim_dns = SSIM(output, target, batch_size)
            ssim_les = SSIM(img, target, batch_size)

            running_loss += loss.item()/len(data_loader)
            avg_psnr += psnr.item()/len(data_loader)
            avg_psnr_les += psnr_les.item()/len(data_loader)
            avg_ssim_dns += ssim_dns.item()/len(data_loader)
            avg_ssim_les += ssim_les.item()/len(data_loader)

        torch.cuda.empty_cache()
        end_time = time.time()

        del img
        del target
        del loss
        del psnr
        del psnr_les

        print("Dev Time: {:.2f} s".format(end_time-start_time))

        return running_loss, avg_psnr_les, avg_psnr, avg_ssim_les, avg_ssim_dns

def test_predictions(model, test_loader, batch_size):
    with torch.no_grad():
        model.eval()
        avg_psnr = 0
        avg_psnr_les = 0
        avg_dns_ke = 0
        avg_les_ke = 0
        avg_recon_ke = 0
        avg_ssim_les = 0
        avg_ssim_dns = 0

        P = []
        L = []
        R = []
        D = []

        L_ = []
        R_ = []
        D_ = []

        for batch_idx, (img, target) in enumerate(test_loader):   
            
            img = img.to(device)
            target = target.to(device)
            
            out = model(img)
            psnr = PSNR(out, target, batch_size)
            psnr_les = PSNR(img, target, batch_size)

            les_ke, recon_ke, dns_ke = KE(img, out, target)
            les_ake, recon_ake, dns_ake = Avg_KE(img, out, target)

            ssim_dns = SSIM(out, target, batch_size)
            ssim_les = SSIM(img, target, batch_size)
            L.append(les_ke.cpu().numpy())
            R.append(recon_ke.cpu().numpy())
            D.append(dns_ke.cpu().numpy())

            L_.append(les_ake.cpu().numpy())
            R_.append(recon_ake.cpu().numpy())
            D_.append(dns_ake.cpu().numpy())

            pic = to_img(out.cpu()) #Only the first image of the batch.
            save_image(pic, 'test_3c/image_{}.png'.format(batch_idx))
            t = to_img(target) #Only the first image of the batch.
            save_image(t, 'test_3c/image_{}_t.png'.format(batch_idx))
            i = to_img(img) #Only the first image of the batch.
            save_image(i, 'test_3c/image_{}_og.png'.format(batch_idx))

            del img
            del target

            avg_psnr += psnr.item()/len(test_loader)
            avg_psnr_les += psnr_les.item()/len(test_loader)

            avg_dns_ke += dns_ke.item()/len(test_loader)
            avg_recon_ke += recon_ke.item()/len(test_loader)
            avg_les_ke += les_ke.item()/len(test_loader)

            avg_ssim_les += ssim_les.item()/len(test_loader)
            avg_ssim_dns += ssim_dns.item()/len(test_loader)

    plot_MAE(L, R, D)
    plot_Avg_MAE(L_, R_, D_)
    return avg_psnr, avg_psnr_les, avg_les_ke, avg_recon_ke, avg_dns_ke, avg_ssim_les, avg_ssim_dns

def train_predictions(model, train_loader):
    with torch.no_grad():
        model.eval()
        avg_psnr = 0
        avg_psnr_les = 0
        avg_dns_ke = 0
        avg_les_ke = 0
        avg_recon_ke = 0
        avg_ssim_les = 0
        avg_ssim_dns = 0

        P = []

        for batch_idx, (img, target) in enumerate(train_loader):  

            # if batch_idx > 3:
            #     break 
            
            img = img.to(device)
            target = target.to(device)
            
            out = model(img)
            psnr = PSNR(out, target, batch_size)
            psnr_les = PSNR(img, target, batch_size)
            les_ke, recon_ke, dns_ke = KE(img, out, target)
            ssim_dns = SSIM(out, target, batch_size)
            ssim_les = SSIM(img, target, batch_size)

            pic = to_img(out.cpu()) #Only the first image of the batch.
            save_image(pic, 'test_3c/image_tr_{}.png'.format(batch_idx))
            t = to_img(target) #Only the first image of the batch.
            save_image(t, 'test_3c/image_tr_{}_t.png'.format(batch_idx))
            i = to_img(img) #Only the first image of the batch.
            save_image(i, 'test_3c/image_tr_{}_og.png'.format(batch_idx))

            del img
            del target

            avg_psnr += psnr.item()/len(train_loader)
            avg_psnr_les += psnr_les.item()/len(train_loader)

            avg_dns_ke += dns_ke.item()/len(train_loader)
            avg_recon_ke += recon_ke.item()/len(train_loader)
            avg_les_ke += les_ke.item()/len(train_loader)

            avg_ssim_les += ssim_les.item()/len(train_loader)
            avg_ssim_dns += ssim_dns.item()/len(train_loader)

    return avg_psnr, avg_psnr_les, avg_les_ke, avg_recon_ke, avg_dns_ke, avg_ssim_les, avg_ssim_dns

