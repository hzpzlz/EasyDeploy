#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-10-29
# @Author  : zhangyuqian@xiaomi.com

import numpy as np
import cv2, os
import scipy.io
from torch import randint
import random
from preproc.raw import pack_raw_to_4ch, rescaling, bayer_to_offsets
from PIL import Image

def image_shift(image,mask,shift):
    '''
    image:  input image to translate
    shift:  max shift pixel value in horizon & vertical
    return: shifted image
    '''

    x = np.random.random_integers(-1*shift,shift,1)
    y = np.random.random_integers(-1*shift,shift,1)

    M = np.float32([[1,0,x],[0,1,y]])
    shifted_image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
    shifted_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),flags=cv2.INTER_NEAREST)
    return shifted_image, shifted_mask

def rotate(image, mask, center):
    (h,w) = image.shape[:2]

    angle = np.random.randint(0,360)

    M = cv2.getRotationMatrix2D((int(center[0]),int(center[1])),angle,scale=1.0)
    rotated_image = cv2.warpAffine(image, M ,(w,h),flags=cv2.INTER_NEAREST)
    rotated_mask = cv2.warpAffine(mask, M, (w,h),flags=cv2.INTER_NEAREST)

    return rotated_image, rotated_mask

def flip(image, mask):

    direction = np.random.randint(3)

    if direction == 0:   #horizontally
        flipped_image = cv2.flip(image, 1)
        flipped_mask = cv2.flip(mask, 1)
    elif direction == 1:
        flipped_image = cv2.flip(image, 0)
        flipped_mask = cv2.flip(mask, 0)
    else:
        # both horizontally and vertically
        flipped_image = cv2.flip(image, -1)
        flipped_mask = cv2.flip(mask ,-1)

    return flipped_image, flipped_mask

def circle_mask(mask,shift):
    ps = mask.shape[0]

    radius = np.random.random_integers(int(ps/16),int(ps/8))   # ps/32   ps/8
    center = np.random.randint(radius+shift+1,ps-radius-shift,2)
    cv2.circle(mask, (center[0],center[1]), radius, (255, 255, 255), -1)

    return mask, center

def rectangle_mask(mask,shift):

    ps = mask.shape[0]
    mask_size = np.random.random_integers(int(ps / 8), int(ps / 4), 2)  # ps/16  ps/4

    pt1_0 = np.random.randint(shift, ps - max(mask_size[0], mask_size[1]) - shift)  # x horizontal
    pt1_1 = np.random.randint(shift, ps - max(mask_size[0], mask_size[1]) - shift)  # y vertically

    cv2.rectangle(mask, (pt1_0, pt1_1), (pt1_0 + mask_size[0], pt1_1 + mask_size[1]), (255, 255, 255), -1)

    center = (pt1_0 + int(mask_size[0]/2), pt1_1 + int(mask_size[1]/2))

    return mask,center

def triangle_mask(mask,shift):
    ps = mask.shape[0]

    position = np.random.randint(2)
    if position == 0:   # left up
        points = np.random.randint(low = shift+1, high = ps/2-shift, size=(3,2))
    elif position == 1: # righ bottom
        points = np.random.randint(low = ps/2+shift, high = ps-shift, size=(3,2))

    edge1 = np.sqrt((points[1][0]-points[0][0])*(points[1][0]-points[0][0])+(points[1][1]-points[0][1])*(points[1][1]-points[0][1]))
    edge2 = np.sqrt((points[2][0]-points[1][0])*(points[2][0]-points[1][0])+(points[2][1]-points[1][1])*(points[2][1]-points[1][1]))
    edge3 = np.sqrt((points[2][0]-points[0][0])*(points[2][0]-points[0][0])+(points[2][1]-points[0][1])*(points[2][1]-points[0][1]))

    while(edge1 + edge2 <= edge3 or edge1 - edge2 >= edge3):
        points = np.random.randint(low=shift + 1, high=ps - shift, size=(3, 2))
        edge1 = np.sqrt((points[1][0] - points[0][0]) * (points[1][0] - points[0][0]) + (points[1][1] - points[0][1]) * (points[1][1] - points[0][1]))
        edge2 = np.sqrt((points[2][0] - points[1][0]) * (points[2][0] - points[1][0]) + (points[2][1] - points[1][1]) * (points[2][1] - points[1][1]))
        edge3 = np.sqrt((points[2][0] - points[0][0]) * (points[2][0] - points[0][0]) + (points[2][1] - points[0][1]) * (points[2][1] - points[0][1]))

    cv2.fillConvexPoly(mask,points,(255,255,255),lineType=cv2.LINE_AA)
    # cv2.fillPoly(mask, points, (255, 255, 255), -1)

    center = (edge2*points[0]+edge3*points[1]+edge1*points[2])/(edge1+edge2+edge3)

    return mask, center

def ellipse_mask(mask,shift):

    ps = mask.shape[0]

    axeLength = np.random.random_integers(int(ps / 8), int(ps / 4),2)  # ps/16  ps/4
    diameter = max(axeLength[0],axeLength[1])
    center = np.random.randint(diameter + shift + 1, ps - diameter - shift, 2)

    cv2.ellipse(mask,(center[0],center[1]),(axeLength[0],axeLength[1]),angle=0,startAngle=0,endAngle=360,color=(255,255,255),thickness=-1)
    return mask,center

def gen_circle(size):

    res = np.random.normal(0, 0.1, size)
    res = np.clip( res + 0.9, 0, 1)

    return res

def PngShow(src, pattern, path, upscaleTimes=1):

    offsets = bayer_to_offsets(pattern)
    img = pack_raw_to_4ch(src, offsets)
    temp = np.clip(img[:, :, :3] * upscaleTimes, 0, 1)
    temp = Image.fromarray(np.uint8(temp * 255))
    temp.save(path)

def local_shift(frames, gt, frame_num):
    
    size = [10, 10, 4]
    shape = frames[0].shape
    marge = [int(shape[0] * 0.25), int(shape[1] * 0.25)]
    start_location = [random.randint(marge[0], shape[0]-marge[0]), random.randint(marge[1], shape[1]-marge[1])]
    patch = gen_circle(size)

    frames[0][start_location[0]:start_location[0]+size[0], start_location[1]:start_location[1]+size[1],:] += patch
    gt[start_location[0]:start_location[0]+size[0], start_location[1]:start_location[1]+size[1],:] += patch
    
    location = start_location
    step_size = [int((shape[0]-2*marge[0])/(4*frame_num)), int((shape[1]-2*marge[1])/(4*frame_num))]
    for i in range(1, frame_num):
        step = [random.randint( 0, 2*step_size[0]) - step_size[0], random.randint(0, 2*step_size[1]) - step_size[1]]
        location = [location[0] +  step[0], location[1] + step[1]]
        frames[i][location[0]:location[0]+size[0], location[1]:location[1]+size[1],:] += patch
        
    for i in range(frame_num):
        frames[i] = np.clip(frames[i], 0, 1)
    gt = np.clip(gt, 0, 1)

    # name = random.randint(1, 10240)
    # for i, frame in enumerate(frames):
    #     PngShow(frame, 'bggr', os.path.join('/home/ran/桌面/DUMP/test/'+ str(name) +'_add_patch_'+str(i) + ".png"))

    return frames, gt

def global_shift(frames, frame_num):
    shape = frames[0].shape
    for i in range(1, frame_num):
        if random.randint(0, 100) < 50:

            step = [random.randint(1, 7), random.randint(1, 7)]
            tmp = frames[i].copy()
            orient = [random.randint(0,1), random.randint(0,1)]

            if orient[0] == 0 and orient[1] == 0:
                frames[i][:shape[0]-step[0], :shape[1]-step[1], :] = tmp[step[0]:, step[1]:, :]
            elif orient[0] == 0 and orient[1] == 1:
                frames[i][:shape[0]-step[0], step[1]:, :] = tmp[step[0]:, :shape[1]-step[1], :]
            elif orient[0] == 1 and orient[1] == 0:
                frames[i][step[0]:, :shape[1]-step[1], :] = tmp[:shape[0]-step[0], step[1]:, :]
            elif orient[0] == 1 and orient[1] == 1:
                frames[i][step[0]:, step[1]:, :] = tmp[:shape[0]-step[0], :shape[1]-step[1], :]

    return frames

def local_shift_patch(frames, frame_num):
    shape = frames[0].shape
    size = [random.randint(10, int(shape[0] * 0.25)),  random.randint(10, int(shape[1] * 0.25))]
    marge = [int(shape[0] - size[0] -  10), int(shape[1] - size[1] - 10)]        
    for i in range(1, frame_num):
        if random.randint(0,100) < 50:
            step = [  random.randint( 1,  7),  random.randint(1,  7) ]    
            location = [random.randint(0, shape[0]-marge[0]), random.randint(0, shape[1]-marge[1])]
            tmp = frames[i].copy()
            frames[i][location[0]:location[0]+size[0], location[1]:location[1]+size[1],:] =  tmp[location[0]+step[0]:location[0]+step[0]+size[0], location[1]+step[1]:location[1]+step[1]+size[1],:]
    return frames

def local_motion(frames, frame_num, motion_shift, ch=4):
    '''
    frames:        packed patch for train
    frame_num :
    trans_frames:  rotated/shifted output
    '''

    # trans_frames = np.zeros_like(frames)
    # trans_frames[:,:,:4] = frames[:,:,:4]
    trans_frames = []
    trans_frames.append(frames[0])
    shift = motion_shift  #10
    frame = frames[0]

    # print(frame.shape)

    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    shape = np.random.randint(4)
    if shape == 0:   # circle
        mask, center = circle_mask(mask, shift)
    elif shape == 1: # rectangle
        mask, center = rectangle_mask(mask, shift)
    elif shape == 2: # ellipse
        mask, center = ellipse_mask(mask, shift)
    elif shape == 3: # triangle
        mask, center = triangle_mask(mask, shift)

    mask_repeat = np.repeat((mask / 255.0)[..., np.newaxis], ch, 2)
    translate = np.random.randint(2)

    for i in range(frame_num-1):

        image = frames[i+1]
        # image = frames[:,:,(i+1)*4:(i+2)*4]
        matte = image * mask_repeat

        # translate
        if translate == 0:    #image_shift
            translate_matte, translate_mask = image_shift(matte, mask, shift)
        else:                 #image_rotation
            translate_matte, translate_mask = rotate(matte, mask, center)

        translate_mask_repeat = np.repeat((translate_mask/255.0)[...,np.newaxis],ch,2)

        trans_image = image * (1-translate_mask_repeat) + translate_matte

        trans_frames.append(trans_image)

    # name = random.randint(1, 10240)
    # for i, frame in enumerate(trans_frames):
    #     PngShow(frame, 'bggr', os.path.join('/home/ran/桌面/DUMP/test/'+ str(name) +'_localmotion_'+str(i) + ".png"))


    return trans_frames

def local_repeat(frames, frame_num):

    base_frame = frames[0]
    trans_frames = []
    trans_frames.append(frames[0])

    # base_frame = frames[:,:,0:4].copy()
    # trans_frames = np.zeros_like(frames)
    # trans_frames[:, :, :4] = frames[:, :, :4]

    mask = np.zeros((base_frame.shape[0], base_frame.shape[1]), dtype=np.uint8)

    shape = np.random.randint(4)
    if shape == 0:  # circle
        mask, center = circle_mask(mask, shift=0)
    elif shape == 1:  # rectangle
        mask, center = rectangle_mask(mask, shift=0)
    elif shape == 2:  # ellipse
        mask, center = ellipse_mask(mask, shift=0)
    elif shape == 3:  # triangle
        mask, center = triangle_mask(mask, shift=0)

    mask_repeat = np.repeat((mask / 255.0)[..., np.newaxis], 4, 2)

    for i in range(frame_num-1):
        image = frames[i+1]
        trans_frames.append(base_frame * mask_repeat + image * (1 - mask_repeat))

     #   print( (image - trans_frames[-1]).mean() )

        # image = frames[:,:,(i+1)*4:(i+2)*4]
        # trans_frames[:,:,(i+1)*4:(i+2)*4] = base_frame * mask_repeat + image * (1-mask_repeat)


    #name = random.randint(1, 10240)
    #for i, frame in enumerate(trans_frames):
    #    PngShow(frame, 'bggr', os.path.join('/home/ran/桌面/DUMP/test/'+ str(name) +'_localrepeat_'+str(i) + ".png"))

    return trans_frames


