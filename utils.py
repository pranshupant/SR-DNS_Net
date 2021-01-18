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

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), -1, 64, 64)
    return x

def PSNR(op, t, batch_size): 
    mse = torch.sum((t - op) ** 2) 
    #print(mse.size())
    mse /= (batch_size*64*64)

    max_pixel = 1.
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    #print(psnr.size())
    return psnr 

def KE(img, op, t): 
    ke_recon = torch.sum(op ** 2)/op.size()[0] 
    ke_dns = torch.sum(t ** 2)/t.size()[0] 
    ke_les = torch.sum(img ** 2)/img.size()[0] 
 
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

def SSIM(op, t, batch_size):
    ssim = 0 
    #print(op.size(), t.size())
    for i in range(op.size()[0]):
        tar = to_img(t[i])
        out = to_img(op[i])
        # print(out[0,0].size())
        (score, diff) = compare_ssim(out[0,0].cpu().numpy(), tar[0,0].cpu().numpy(), full=True)
        ssim+=score/batch_size
    
        #print("SSIM: {}".format(score))
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

def plot_Avg_MAE(L, R, D):

    # MAE = np.abs((np.array(R) - np.array(D))/np.array(D))
    # print(MAE.shape)
    # print(MAE)
    avg = np.mean(np.abs(R))
    # plt.xticks([])
    plt.title('Reconstruction')
    plt.ylabel('Average Turbulent Velocity')
    # plt.ylim(0,1.2)
    plt.plot(np.abs(R), 'ko', fillstyle = 'none')
    plt.axhline(avg, color = 'r')
    plt.savefig('AM.eps')
    plt.close()

    # MAE_L = np.abs((np.array(L) - np.array(D))/np.array(D))
    # print(MAE.shape)
    # print(MAE)
    avg_l = np.mean(np.abs(D))
    # plt.xticks([])
    plt.title('DNS')
    plt.ylabel('Average Turbulent Velocity')
    # plt.plot(MAE, 'yo', fillstyle = 'none')
    plt.plot(np.abs(D), 'ko', fillstyle = 'none')
    # plt.axhline(avg, color = 'r')
    plt.axhline(avg_l, color = 'r')
    plt.savefig('AM_L.eps')  # plt.show()
    plt.close()

    error = np.abs(np.array(D) - np.array(R))
    avg_e = np.mean(np.abs(error))
    # plt.xticks([])
    plt.title('Error')
    plt.ylim(0, 0.35)
    plt.ylabel('Turbulent Velocity Error')
    # plt.plot(MAE, 'yo', fillstyle = 'none')
    plt.plot(error, 'ko', fillstyle = 'none')
    # plt.axhline(avg, color = 'r')
    plt.axhline(avg_e, color = 'r')
    plt.savefig('error.eps')  # plt.show()
    plt.close()

#combined plot for turbulent velocity
def plot_Avg_MAE(L, R, D):

    # MAE = np.abs((np.array(R) - np.array(D))/np.array(D))
    # print(MAE.shape)
    # print(MAE)
    avg = np.mean(np.abs(R))
    # plt.xticks([])
    # plt.title('Reconstruction')
    plt.ylabel('Average Turbulent Velocity')
    plt.xlabel('Samples')
    # plt.ylim(0,1.2)
    plt.plot(np.abs(R),  marker = '^', c = 'dodgerblue', label='Recon',ls=' ', ms='3.5')


    # MAE_L = np.abs((np.array(L) - np.array(D))/np.array(D))
    # print(MAE.shape)
    # print(MAE)
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