#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cv2
import os
from PIL import Image

#img1 = cv2.imread('/home/hzp/codes/BasicAlign/result/pwcplus_onnx/rawdata/frame_0016.png').astype(np.float32)[:,:,[2,1,0]]
#img2 = cv2.imread('/home/hzp/codes/BasicAlign/result/pwcplus_onnx/rawdata/frame_0017.png').astype(np.float32)[:,:,[2,1,0]]
img1 = Image.open('/home/hzp/codes/BasicAlign/result/pwcplus_onnx/rawdata/frame_0016.png')
img2 = Image.open('/home/hzp/codes/BasicAlign/result/pwcplus_onnx/rawdata/frame_0017.png')

img1 = np.array(img1).astype(np.float32)
img2 = np.array(img2).astype(np.float32)

output_path = 'rawdata_bin'
if not os.path.exists(output_path):
    os.makedirs(output_path)

fid1 = open(output_path + '/input1.raw', 'wb')
img1.tofile(fid1)
fid1.close()

fid2 = open(output_path + '/input2.raw', 'wb')
img2.tofile(fid2)
fid2.close()

if os.path.exists('rawdata.txt'):
    os.remove('rawdata.txt')

with open('rawdata.txt', 'a') as f:
    f.write("input1:="+output_path+'/input1.raw' ' ' "input2:="+output_path+'/input2.raw')


