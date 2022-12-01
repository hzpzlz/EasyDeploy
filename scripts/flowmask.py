import os
import time
import argparse
import glob, cv2
import numpy as np
import math
from numba import jit

import imageio
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root_path',type=str,default='/home/hzp/datasets/night_scan/raw_base_align')
parser.add_argument('--result_path',type=str,default='/home/hzp/datasets/night_scan/raw_base_align')
args=parser.parse_args()

def convertOptRGB(optical_flow, OutFolder, basename):
    '''
    :param optical_flows:  h*w*2
    :param OutFolder:
    :param basename:
    :return:
    '''

    #for i, optical_flow in enumerate(optical_flows):
        #flow_img = flow_viz.flow_to_image(optical_flow)

    optical_flow = optical_flow.transpose(2, 0, 1)
    blob_x = optical_flow[0]
    blob_y = optical_flow[1]

    hsv = np.zeros((blob_x.shape[0], blob_y.shape[1], 3), np.uint8)
    mag, ang = cv2.cartToPolar(blob_x, blob_y)
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = np.zeros((blob_x.shape[0], blob_x.shape[1],3),np.uint8)
    result[:, :, 0] = bgr[:, :, 2]
    result[:, :, 1] = bgr[:, :, 1]
    result[:, :, 2] = bgr[:, :, 0]

    cv2.imwrite('flow' + basename + '.jpg', result)

def rgb2gray(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return gray

def getflow(gray1, gray2):
    dis = cv2.DISOpticalFlow_create(2)
    flow = dis.calc(gray1, gray2, None, )
    return flow

def cvremap(img, flow):
    h,w = flow.shape[:2]
    X, Y = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))

    map_x = (X + flow[:, :, 0:1].squeeze()).astype(np.float32)
    map_y = (Y + flow[:, :, 1:2].squeeze()).astype(np.float32)

    warp = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return warp
  
@jit  
def getflowmask(flow, flow2):
    h,w = flow.shape[:2]
    flowwarp = cvremap(flow2, flow)
    
    flow_sum = flow+flowwarp
    convertOptRGB(flow_sum, '', 'fbsum')
    mask = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            tmp = flow_sum[i,j,0]*flow_sum[i,j,0] + flow_sum[i,j,1]*flow_sum[i,j,1]
            if(tmp>25):
                mask[i,j]=1
            else:
                mask[i,j]=0
            
    return mask

if __name__ == "__main__":
    img1 = cv2.imread('input_ev0_1.png')
    img2 = cv2.imread('input_ev0_2.png')
    gray1 = rgb2gray(img1)
    gray2 = rgb2gray(img2)
    
    flow = getflow(gray1, gray2)
    convertOptRGB(flow, '', 'ford')
    flow2 = getflow(gray2, gray1)
    convertOptRGB(flow2, '', 'back')
    warp = cvremap(img2, flow)
    
    mask = getflowmask(flow, flow2)
    maskwarp = img1*mask[:,:, np.newaxis] + (1-mask[:,:, np.newaxis])*warp
    
    cv2.imwrite('warp.jpg', warp)
    cv2.imwrite('mask.jpg', mask*255)
    cv2.imwrite('warp_mask.jpg', maskwarp)
    
    
    