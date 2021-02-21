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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## TODO : [*] Remove batch_size decl.
## TODO : [*] Print within the fucntions not main
## TODO : [] Fix test predictions with list averaging

def train_epoch(model, data_loader, criterion, optimizer):
    model.train()
    
    running_loss = []
    avg_psnr = []
    avg_psnr_les = []

    start_time = time.time()
    print('Train Loop')

    for batch_num, (img, target) in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True):

        img = img.to(device)
        target = target.to(device)

        output = model(img)
        loss = criterion(output, target)
        psnr = PSNR(output, target)
        psnr_les = PSNR(img, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        avg_psnr.append(psnr.item())
        avg_psnr_les.append(psnr_les.item())

    #print(' ')
    print(f'Train_Loss:{sum(running_loss)/len(running_loss):.6f}, PSNR_DNS:{sum(avg_psnr)/len(avg_psnr):.4f}, PSNR_LES:{sum(avg_psnr_les)/len(avg_psnr_les):.4f}')
    torch.cuda.empty_cache()
    end_time = time.time()
    del img
    del target
    del loss
    print("Train Time: {:.2f} s".format(end_time-start_time))

    return sum(running_loss)/len(running_loss)


def dev_epoch(model, data_loader, criterion):
    with torch.no_grad():
        model.eval()

        running_loss = []
        avg_psnr = []
        avg_psnr_les = []
        avg_ssim_dns = []
        avg_ssim_les = []

        start_time = time.time()
        print('Dev Loop')
        print(' ')

        for batch_num, (img, target) in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True):

            img = img.to(device)
            target = target.to(device)

            output = model(img)
            loss = criterion(output, target)
            psnr = PSNR(output, target)
            psnr_les = PSNR(img, target)
            ssim_dns = SSIM(output, target)
            ssim_les = SSIM(img, target)

            running_loss.append(loss.item())
            avg_psnr.append(psnr.item())
            avg_psnr_les.append(psnr_les.item())
            avg_ssim_dns.append(ssim_dns.item())
            avg_ssim_les.append(ssim_les.item())

        torch.cuda.empty_cache()
        end_time = time.time()

        del img
        del target
        del loss
        del psnr
        del psnr_les

        print(f"Dev Time: {end_time-start_time:.2f} s")
        print(f'PSNR_DNS:{sum(avg_psnr)/len(avg_psnr):.4f}, PSNR_LES:{sum(avg_psnr_les)/len(avg_psnr_les):.4f}, SSIM_DNS:{sum(avg_ssim_dns)/len(avg_ssim_dns):.4f}, SSIM_LES:{sum(avg_ssim_les)/len(avg_ssim_les):.4f}')

        return sum(running_loss)/len(running_loss)

def test_predictions(model, test_loader, root):
    with torch.no_grad():
        model.eval()
        avg_psnr = 0
        avg_psnr_les = 0
        avg_dns_ke = 0
        avg_les_ke = 0
        avg_recon_ke = 0
        avg_ssim_les = 0
        avg_ssim_dns = 0

        L = []
        R = []
        D = []

        L_ = []
        R_ = []
        D_ = []

        for batch_idx, (img, target) in enumerate(test_loader):   
            
            img = img.to(device)
            target = target.to(device)
            
            # t1 = time.time()
            out = model(img)
            # print(f'Avg Time for evaluation: {(time.time()-t1)*(1/1000)} secs')

            psnr = PSNR(out, target)
            psnr_les = PSNR(img, target)

            les_ke, recon_ke, dns_ke = KE(img, out, target)
            les_ake, recon_ake, dns_ake = Avg_KE(img, out, target)

            ssim_dns = SSIM(out, target)
            ssim_les = SSIM(img, target)
            L.append(les_ke.cpu().numpy())
            R.append(recon_ke.cpu().numpy())
            D.append(dns_ke.cpu().numpy())

            L_.append(les_ake.cpu().numpy())
            R_.append(recon_ake.cpu().numpy())
            D_.append(dns_ake.cpu().numpy())

            pic = to_img(out) #Only the first image of the batch.
            save_image(pic, f'{root}/test_3c/image_{batch_idx}.png')
            t = to_img(target) #Only the first image of the batch.
            save_image(t, f'{root}/test_3c/image_{batch_idx}_t.png')
            i = to_img(img) #Only the first image of the batch.
            save_image(i, f'{root}/test_3c/image_{batch_idx}_og.png')

            del img
            del target

            avg_psnr += psnr.item()/len(test_loader)
            avg_psnr_les += psnr_les.item()/len(test_loader)

            avg_dns_ke += dns_ke.item()/len(test_loader)
            avg_recon_ke += recon_ke.item()/len(test_loader)
            avg_les_ke += les_ke.item()/len(test_loader)

            avg_ssim_les += ssim_les.item()/len(test_loader)
            avg_ssim_dns += ssim_dns.item()/len(test_loader)

    plot_MAE(root, L, R, D)
    plot_Avg_MAE(root, L_, R_, D_)

    print(f"Metrics On Test Data")
    print(f"PSNR_RECON: {avg_psnr:.4f},PSNR_LES: {avg_psnr_les:.4f}")
    print(f"KE_RECON: {avg_dns_ke:.4f}, KE_LES: {avg_les_ke:.4f}, KE_RECON: {avg_recon_ke:.4f}")
    print(f"SSIM_RECON: {avg_ssim_dns:.4f}, SSIM_LES: {avg_ssim_les:.4f}")

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

        for batch_idx, (img, target) in enumerate(train_loader):  
            
            img = img.to(device)
            target = target.to(device)
            
            out = model(img)
            psnr = PSNR(out, target)
            psnr_les = PSNR(img, target)
            les_ke, recon_ke, dns_ke = KE(img, out, target)
            ssim_dns = SSIM(out, target)
            ssim_les = SSIM(img, target)

            pic = to_img(out) #Only the first image of the batch.
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

    print(f"Metrics On Train Data")
    print(f"PSNR_RECON: {avg_psnr:.4f},PSNR_LES: {avg_psnr_les:.4f}")
    print(f"KE_RECON: {avg_dns_ke:.4f}, KE_LES: {avg_les_ke:.4f}, KE_RECON: {avg_recon_ke:.4f}")
    print(f"SSIM_RECON: {avg_ssim_dns:.4f}, SSIM_LES: {avg_ssim_les:.4f}")

