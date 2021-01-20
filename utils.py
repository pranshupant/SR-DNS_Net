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
from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(3, 64, 64)
    return x

def create_directories(root, mode):

    if not os.path.exists(f'{root}/model{mode}'):
        os.mkdir(f'{root}/model{mode}')

    if not os.path.exists(f'{root}/test{mode}'):
        os.mkdir(f'{root}/test{mode}')

# def PSNR(op, t, batch_size): 
#     mse = torch.sum((t - op) ** 2) 
#     #print(mse.size())
#     mse /= (batch_size*64*64)

#     max_pixel = 1.
#     psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
#     #print(psnr.size())
#     return psnr 

def PSNR(op, t):
    batch_size = op.shape[0]
    psnr = sum([peak_signal_noise_ratio(to_img(op[i]).cpu().detach().numpy(), to_img(t[i]).cpu().detach().numpy()) for i in range(op.shape[0])])/batch_size
    #print(psnr.size())
    return psnr 

def KE(img, op, t): 
    ke_recon = torch.sum(op ** 2)/op.shape[0] 
    ke_dns = torch.sum(t ** 2)/t.shape[0] 
    ke_les = torch.sum(img ** 2)/img.shape[0] 
 
    return ke_les, ke_recon, ke_dns

def Avg_KE(img, op, t): 
    # pdb.set_trace()

    op = np.squeeze(op)
    img = np.squeeze(img)
    t = np.squeeze(t)
    # pdb.set_trace()

    ke_recon = torch.mean(torch.abs(op - torch.mean(op)))
    ke_dns = torch.mean(torch.abs(t - torch.mean(t)))
    ke_les = torch.mean(torch.abs(img - torch.mean(img)))
 
    return ke_les, ke_recon, ke_dns

def SSIM(op, t):
    batch_size = op.shape[0]
    ssim = sum([structural_similarity(to_img(op[i]).cpu().detach().numpy().transpose(1,2,0), to_img(t[i]).cpu().detach().numpy().transpose(1,2,0),
                multichannel=True) for i in range(op.shape[0])])/batch_size

    return ssim

def plot_MAE(L, R, D):

    MAE = np.abs((np.array(R) - np.array(D))/np.array(D))
    print(MAE.shape)
    # print(MAE)
    avg = np.mean(MAE)
    plt.xticks([])
    plt.title('Reconstruction')
    plt.ylabel('MAE')
    plt.ylim(0,1.2)
    plt.plot(MAE, 'ko', fillstyle = 'none')
    plt.axhline(avg, color = 'r')
    plt.savefig('MAE.eps')
    plt.close()

    MAE_L = np.abs((np.array(L) - np.array(D))/np.array(D))
    print(MAE.shape)
    # print(MAE)
    avg_l = np.mean(MAE_L)
    plt.xticks([])
    plt.title('LES')
    plt.ylabel('MAE')
    # plt.plot(MAE, 'yo', fillstyle = 'none')
    plt.plot(MAE_L, 'ko', fillstyle = 'none')
    # plt.axhline(avg, color = 'r')
    plt.axhline(avg_l, color = 'r')
    plt.savefig('MAE_L.eps')  # plt.show()
    plt.close()

#combined plot for turbulent velocity
def plot_Avg_MAE(L, R, D):

    avg = np.mean(np.abs(R))
    # plt.xticks([])
    # plt.title('Reconstruction')
    plt.ylabel('Average Turbulent Velocity')
    plt.xlabel('Samples')
    # plt.ylim(0,1.2)
    plt.plot(np.abs(R),  marker = '^', c = 'dodgerblue', label='Recon',ls=' ', ms='3.5')


    avg_l = np.mean(np.abs(D))
    # plt.xticks([])
    # plt.title('DNS')
    # plt.ylabel('Average Turbulent Velocity')
    # plt.plot(MAE, 'yo', fillstyle = 'none')
    plt.plot(np.abs(D), marker = '+', c = 'darkorange', label='DNS',ls=' ', ms='4')
    # plt.axhline(avg, color = 'r')
    plt.axhline(avg_l, color = 'red', ls='-.',label='Avg. DNS', lw='1.5')
    plt.axhline(avg, color = 'k', ls='-.', label='Avg. Recon', lw='1.5')
    plt.legend(loc='best')
    plt.savefig('combined.eps')  # plt.show()
    plt.close()