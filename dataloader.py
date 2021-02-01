import os
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image
import time
import pdb
from PIL import Image
import matplotlib.pyplot as plt 

## TODO: [] np.permute for selecting sample ids.


class MyDataset(data.Dataset):
    def __init__(self, mode, root, img_list, data_split, transform=None):

        self.ROOT = root

        self.LES_DATA_PATH = self.ROOT + "/DNS-LES_128_3C/les_3c/"
        self.DNS_DATA_PATH = self.ROOT + "/DNS-LES_128_3C/dns_3c/"

        self.img_list = img_list

        self.data_split = data_split

        if mode == 'train':
            self.image_paths, self.target_paths = self.get_train_data()

        elif mode == 'dev':
            self.image_paths, self.target_paths = self.get_dev_data()

        elif mode == 'test':
            self.image_paths, self.target_paths = self.get_test_data()

        else:
            print("Incorrect Mode!")

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        ip = Image.open(self.image_paths[index])
        op = Image.open(self.target_paths[index])
        x = self.transform(ip)
        y = self.transform(op)
        return x, y

    def get_train_data(self):

        nums = self.img_list[:int(self.data_split[0])]
        image_paths = [self.LES_DATA_PATH+str(i/5670)[:3]+"0/"+str(i%567)+".png" 
                        for i in nums]
        target_paths = [self.DNS_DATA_PATH+str(i/5670)[:3]+"0/"+str(i%567)+".png" 
                        for i in nums] 


        print(f'Training Data Samples: {len(image_paths)}')
        # print(image_paths[39999])
        return image_paths, target_paths


    def get_dev_data(self):

        nums = self.img_list[-(int(self.data_split[1])+int(self.data_split[2])): -int(self.data_split[2])]
        image_paths_dev = [self.LES_DATA_PATH+str(i/5670)[:3]+"0/"+str(i%567)+".png" 
                            for i in nums]
        target_paths_dev = [self.DNS_DATA_PATH+str(i/5670)[:3]+"0/"+str(i%567)+".png" 
                            for i in nums] 
        print(f'Development Data Samples: {len(image_paths_dev)}')
    
        # print(image_paths_dev[0])
        return image_paths_dev, target_paths_dev


    def get_test_data(self):

        nums = self.img_list[-int(self.data_split[2]):]
        image_paths_test = [self.LES_DATA_PATH+str(i/5670)[:3]+"0/"+str(i%567)+".png" 
                            for i in nums]
        target_paths_test = [self.DNS_DATA_PATH+str(i/5670)[:3]+"0/"+str(i%567)+".png" 
                            for i in nums] 
        
        print(f'Test Data Samples: {len(image_paths_test)}')
    
        # print(image_paths_test[0])  
        return image_paths_test, target_paths_test