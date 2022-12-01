from utils.raw_io import *
from utils.raw import *

import glob
import os
import numpy as np

import argparse

def convertRawToJpg(img_path, ext, idx, ratio, ori, flag, save_path):
    if (ext=="idealraw" or  ext=="GBRG" or ext=='*.finalmask' or ext=='*.dgmask' or ext=='*.flowmask' or ext=='*.dgflowmask') and ori=="UINT8":
        args.inheight = 768
        args.inwidth = 1024
        noise_img = laod_uint8(img_path, args.inwidth, args.inheight)
        #np.savetxt(os.path.join(save_path,'mask_'+str(idx)+'_hzp.csv'), noise_img, delimiter = ',')

        save_noise = np.zeros((args.inheight , args.inwidth))
        save_noise = noise_img
        noise_tmp = Image.fromarray(np.uint8(save_noise))
        noise_tmp.save(os.path.join(save_path, 'flowdgmask_' + str(idx) + flag + '.jpg'))

    if (ext=="idealraw" or  ext=="GBRG" or ext=='*.gray') and ori=="UINT8":
        args.inheight = 1536
        args.inwidth = 2048
        noise_img = laod_uint8(img_path, args.inwidth, args.inheight)
        #np.savetxt(os.path.join(save_path,'mask_'+str(idx)+'_hzp.csv'), noise_img, delimiter = ',')

        save_noise = np.zeros((args.inheight , args.inwidth))
        save_noise = noise_img
        noise_tmp = Image.fromarray(np.uint8(np.clip(save_noise, 0, 255)))
        noise_tmp.save(os.path.join(save_path, 'mask_' + str(idx) + flag + '.jpg'))

    if (ext=="idealraw" or  ext=="GBRG" or ext=='*.homoraw' or ext=='*.cluma' or ext=='*.deghostmask' or ext=='*.twmask') and ori=="UINT8":
        args.inheight = 96
        args.inwidth = 128
        noise_img = laod_uint8(img_path, args.inwidth, args.inheight)
        #np.savetxt(os.path.join(save_path,'mask_'+str(idx)+'_.csv'), noise_img, delimiter = ',')

        save_noise = np.zeros((args.inheight , args.inwidth))
        save_noise = noise_img
        noise_tmp = Image.fromarray(np.uint8(np.clip(save_noise, 0, 255)))
        noise_tmp.save(os.path.join(save_path, 'gray_' + str(idx) + flag + '.jpg'))

    if (ext=="idealraw" or  ext=="GBRG" or ext=='*.disraw') and ori=="UINT8":
        args.inheight = 1536
        args.inwidth = 2048
        noise_img = laod_uint8(img_path, args.inwidth, args.inheight)
        #np.savetxt(os.path.join(save_path,'mask_'+str(idx)+'_hzp.csv'), noise_img, delimiter = ',')

        save_noise = np.zeros((args.inheight , args.inwidth))
        save_noise = noise_img
        noise_tmp = Image.fromarray(np.uint8(save_noise))
        noise_tmp.save(os.path.join(save_path, 'gray_' + str(idx) + flag + '.jpg'))


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root_path',type=str,default='')
parser.add_argument('--inheight', type=int, default=3072, help='height of input raw images')
parser.add_argument('--inwidth', type=int, default=4080, help='width of input raw images')
parser.add_argument('--black_level', type=int, default=0, help='black level of input data')
parser.add_argument('--white_level', type=int, default=255, help='white level of input data')
parser.add_argument('--target_pattern', type=str, default='gbrg', help='')
parser.add_argument('--pack_pattern', type=str, default='rgbg', help='')
parser.add_argument('--ext', type=str, default='*.raw', help='')
parser.add_argument('--ori', type=str, default='UINT8', help='')
parser.add_argument('--flag', type=str, default='', help='')
args=parser.parse_args()
flag = args.flag
root_path = args.root_path
ext = args.ext
ori = args.ori

ratio = 10
if ratio>60:
    ratio=60

for root_dir in os.listdir(root_path):
    file_path = os.path.join(root_path, root_dir)
    img_path = sorted(glob.glob(os.path.join(file_path, ext))) #
    for i in range(len(img_path)):
        convertRawToJpg(img_path[i], ext, i, ratio, ori, flag, file_path)

