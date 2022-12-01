from __future__ import division
import os, scipy.io
import numpy as np
import glob
import cv2
import scipy.misc
import scipy.io as scio
from PIL import Image
import argparse


def Decode_Plain(rawPath, width, height):

    file = open(rawPath, 'rb')
    rawdata = file.read()
    file.close()

    img = Image.frombytes('I;16', (width, height), rawdata)
    img = np.uint16(img)

    return img



def Encode_Plain(raw16bit):

    raw16bit = np.uint16(raw16bit)

    img_bin = Image.fromarray(raw16bit, 'I;16').tobytes()

    return img_bin

def Convert_Plain(raw16bit, width, height):

    raw16bit = np.uint16(raw16bit)

    second_byte = np.right_shift(raw16bit, 8)
    data_per_line = int(width * 2)
    MIPI_10BIT = np.zeros([height, data_per_line])
    MIPI_10BIT[:, 0:data_per_line:2] = np.uint8(raw16bit)
    MIPI_10BIT[:, 1:data_per_line:2] = np.uint8(second_byte)

    return MIPI_10BIT

def unpack_raw(image):

    H=image.shape[0]
    W=image.shape[1]
    raw=np.zeros((H*2,W*2),dtype=image.dtype)

    raw[0:H * 2:2, 0:W * 2:2] = image[:, :, 1]
    raw[0:H * 2:2, 1:W * 2:2] = image[:, :, 0]
    raw[1:H * 2:2, 0:W * 2:2] = image[:, :, 2]
    raw[1:H * 2:2, 1:W * 2:2] = image[:, :, 3]   #RGGB->GRBG

    raw = raw.astype(np.uint16)

    return raw

def read_plainraw(rawPath, width, height, stride):

    raw16bit = np.zeros([height, width])
    file = open(rawPath, 'rb')
    rawdata = file.read()
    file.close()

    stride = (width * 2)

    rawdata = Image.frombytes('L', (int(stride), int(height)), rawdata)
    rawdata = np.asarray(rawdata)

    data_per_line = int(width * 2)
    first_byte = np.uint16(rawdata[:, 0:data_per_line:2])
    second_byte = np.uint16(rawdata[:, 1:data_per_line:2])

    # get 10bit data
    first_pixel = np.bitwise_or(np.left_shift(second_byte, 8), first_byte)
    # first_pixel = np.bitwise_or(np.left_shift(first_byte, 8), second_byte)

    raw16bit[:, 0:width] = first_pixel

    return raw16bit

def rgbg2bayer(raw):
    height, width = raw.shape[0:2]
    raw_unpack = np.zeros([height, width])
    raw = raw.reshape([int(height/2), int(width/2), 4])
    # RGBG  TO  BG
    #           GR
    raw_unpack[::2,::2] = raw[:,:,2]
    raw_unpack[::2, 1::2] = raw[:, :, 1]
    raw_unpack[1::2, ::2] = raw[:, :, 3]
    raw_unpack[1::2, 1::2] = raw[:, :, 0]
    return raw_unpack

def pack_raw(raw_im):
    # pack bayer image into 4 channels, raw_im is an Image object
    im = np.float32(raw_im)
    # im = np.maximum(im - 64, 0) / (1023 - 64)  # subtract the black level
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :]), axis=2)
    return out


def get_ratio_K1(filename):
    _, filename = os.path.split(filename)
    cl = float((filename.split('_cluma[')[-1]).split(']_')[0])
    tl = float((filename.split('_tluma[')[-1]).split(']_')[0])
    predg = float((filename.split('_ispdg[')[-1]).split(']_')[0])
    spre = float((filename.split('_rtexp[')[-1]).split(']_')[0])
    scap = float((filename.split('_capexp[')[-1]).split(']_')[0])
    ratio = tl / cl * predg * (spre / scap)
    return ratio

def get_ratio(filename):
    cl = float((filename.split('_CL[')[-1]).split(']_')[0])
    tl = float((filename.split('_TL[')[-1]).split(']_')[0])
    predg = float((filename.split('_PREDG[')[-1]).split(']_')[0])
    spre = float((filename.split('_SPRE[')[-1]).split(']_')[0])
    scap = float((filename.split('_SCAP[')[-1]).split(']_')[0])
    ratio = tl / cl * predg * (spre / scap)
    return ratio

def show_grbg(data_path, output_path, black_l = 1024, white_l = 16383, suffix = '.RawPlain16LSB1', flag='', h = 3000, w = 4000, ratio = 1, oriraw=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filelist = os.listdir(data_path)
    filelist.sort()
    count = 0
    for filename in filelist:
        ratio_temp = ratio
        if filename.endswith(suffix):
            print(filename, 'ffffffffffffffffffffffffffffffffffffffffff')
            filepath = os.path.join(data_path, filename)
            #if 'result_0.alignraw' in filename:
            #     ratio_temp = 8 * ratio_temp
            #elif 'result_1.alignraw' in filename:
            #     ratio_temp = 2 * ratio_temp

            print('final ratio is: ', ratio_temp)

            # if not '_output_' in filename:
            #     continue
            # ratio_temp = 1
            # if 'req[13]' in filename:
            #     ratio_temp = 1
            # if 'req[16]' in filename:
            #     ratio_temp = 1
            output_raw_ = read_plainraw(filepath, w, h, 0)
            # print(np.max(output_raw_))
            # print(np.min(output_raw_))
            output_raw_ = output_raw_.astype(np.float)
            output_raw = np.minimum(np.maximum((output_raw_ - black_l), 0), (white_l - black_l)) / (
                    white_l - black_l)
            if oriraw==True:
                tmp = rgbg2bayer(output_raw)
                output_raw = tmp

            print(np.mean(output_raw), 'mmmmm')
            output = pack_raw(output_raw)
            gt3 = np.zeros((output.shape[0], output.shape[1], 3))
            gt3[:, :, 0] = output[:, :, 0]
            gt3[:, :, 1] = (output[:, :, 1] + output[:, :, 2]) / 2
            gt3[:, :, 2] = output[:, :, 3]

            gt3 = gt3 * 255 * ratio_temp
            gt3 = gt3.clip(0, 255)
            gt3 = gt3.astype(np.uint8)


            #cv2.imwrite(os.path.join(output_path, 'RGB_' + filename.split('.')[0] + '_' + flag + '.jpg'), gt3)
            cv2.imwrite(os.path.join(output_path, 'RGB_' + str(count) + '_' + flag + '.jpg'), gt3)
            count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--root_path',type=str,default='/home/hzp/datasets/night_scan/raw_base_align')
    parser.add_argument('--result_path',type=str,default='/home/hzp/datasets/night_scan/raw_base_align')
    parser.add_argument('--ext',type=str,default='/home/hzp/datasets/night_scan/raw_base_align')
    parser.add_argument('--flag',type=str,default='/home/hzp/datasets/night_scan/raw_base_align')
    parser.add_argument('--ratio',type=float,default=1.0)
    parser.add_argument('--oriraw',type=bool,default=False)

    args=parser.parse_args()
    black_l = 1024
    white_l = 16383
    h = 3072
    w = 4096
    # ratio = 'K1'
    #path = '/home/hzp/datasets/night_scan/problem/partdatares_dego2'
    path = args.root_path
    output_path = path
    ext = args.ext
    flag = args.flag
    ratio = args.ratio
    oriraw = args.oriraw
    # show_grbg(path, output_path, black_l, white_l, '.RGBG4', h, w, 5)
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir == 'preview':
                continue
            # if not ('input' in dir and 'ratio' in root):
            #     continue
            # if not 'ratio' in dir:
            # #     continue
            #ratio = 10
            if 'pre_input' in root:
                continue
            if 'pre_output' in root:
                continue
            output_path_1 = os.path.join(root, dir)
            #print(output_path_1, 'oooo')
            show_grbg(os.path.join(root, dir), output_path_1, black_l, white_l, ext, flag, h, w, ratio, oriraw)

    #chmod777(path)
    
