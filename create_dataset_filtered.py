import numpy as np
import shutil
import pyJHTDB
import pyJHTDB.dbinfo
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import time
import argparse
from multiprocessing import Pool

# sample command: python3 create_dataset_filtered.py -k 21 -root data/k21
# sample command: python3 create_dataset_filtered.py -k 11 -root data/k11
# sample command: python3 create_dataset_filtered.py -k 41 -root data/k41

N = 128
p = np.linspace(0, 1, N)
X = np.linspace(0, 2*np.pi, 4)
Y = np.linspace(0, 2*np.pi, 4)
Z = np.arange(0, 2*np.pi, 0.1)

x = np.zeros((N, N, 3), np.float32)

LJHTDB = pyJHTDB.libJHTDB()
LJHTDB.initialize()
 
#This is the token assigned to Pranshu Pant
# LJHTDB.add_token(str(os.environ.get('LJHTDB_auth_token')))
LJHTDB.add_token("edu.cmu.andrew.ppant-68a123d6")

def u_data(time, x, k):

    # u = LJHTDB.getData(
    #             time,
    #             x,
    #             sinterp = 4,
    #             getFunction='getVelocity')
    ubox = LJHTDB.getBoxFilter(
                time,
                x,
                field = 'velocity',
                filter_width = k*(2*np.pi / 1024))
    
    return ubox

def create_turb_dataset(t, k, root):
    count = 0

    start_time = time.time()
    print(f"Time:",t)
    for idx in range(len(X)-1):
        px = np.linspace(X[idx], X[idx+1], N)

        for idy in range(len(Y)-1):
            py = np.linspace(Y[idy], Y[idy+1], N)

            for z in Z:
                x[:, :, 0] = px[np.newaxis, :]
                x[:, :, 1] = py[:, np.newaxis]
                x[:, :, 2] = z

                u_box = u_data(t, x, k)

                if not os.path.exists(f'{root}/DNS-LES_128_3C/les/%.2f'%t):
                    os.mkdir(f'{root}/DNS-LES_128_3C/les/%.2f'%t)
                # if not os.path.exists(f'{root}/DNS-LES_128_3C/dns/%.2f'%t):
                #     os.mkdir(f'{root}/DNS-LES_128_3C/dns/%.2f'%t)

                
                # norm1 = cv2.normalize(u, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                norm2 = cv2.normalize(u_box, 0, 255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                
                if count%100 == 0:
                    print(count)

                # cv2.imwrite(f"{root}/DNS-LES_128_3C/dns/%.2f/%d.png"%(t,count), norm1)
                cv2.imwrite(f"{root}/DNS-LES_128_3C/les/%.2f/%d.png"%(t,count), norm2)
                count+=1
                end_time = time.time()
                    
    print("Time: {:.2f} s".format(end_time-start_time))


def create_dirs(root):

    dir1, dir2, dir3 = False, False, False

    if not os.path.exists(f"{root}/DNS-LES_128_3C"):
        os.mkdir(f"{root}/DNS-LES_128_3C")
        dir1 = True

    if not os.path.exists(f"{root}/DNS-LES_128_3C/les"):
        os.mkdir(f"{root}/DNS-LES_128_3C/les")
        dir2 = True

    if not os.path.exists(f"{root}/DNS-LES_128_3C/dns"):
        os.mkdir(f"{root}/DNS-LES_128_3C/dns")
        dir3 = True

    if not (dir1 and dir2 and dir3):
        print('Dataset Already Created')
        return True
    else:  
        print('Created Directories')
        return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', dest='k', default='11', help="Select the filter size for box filter")
    parser.add_argument('-root', dest='root', help="Select the directory where you want to save the dataset")
    args = parser.parse_args()

    run = create_dirs(args.root)

    if run:
        T = [7.7]
        # T = np.arange(9.2, 10.1, 0.1)
        iter_T = {(t, int(args.k), args.root) for t in T}

        y = Pool()
        y.starmap(create_turb_dataset, iter_T)

