#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-11-27
# @Author  : zhangyuqian@xiaomi.com

import numpy as np
from PIL import Image

def pack_raw(raw_im, max_value, black_level):
    # pack bayer image into 4 channels, raw_im is an Image object
    im = np.float32(raw_im)
    # im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    im = np.maximum(im - black_level, 0) / (max_value - black_level)  # subtract the black level
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    # out = np.concatenate((im[0:H:2, 0:W:2, :],
    #                      im[0:H:2, 1:W:2, :],
    #                      im[1:H:2, 1:W:2, :],
    #                      im[1:H:2, 0:W:2, :]), axis=2)  #Sony: RGBG->RGBG

    out = np.concatenate((im[0:H:2, 1:W:2, :],
                          im[0:H:2, 0:W:2, :],
                          im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :]), axis=2)  # MIphone: GRBG->RGBG

    # out = np.concatenate((im[0:H:2, 1:W:2, :],
    #                       im[0:H:2, 0:W:2, :],
    #                       im[1:H:2, 1:W:2, :],
    #                       im[1:H:2, 0:W:2, :]), axis=2) #MIphone: GRBG->RGGB
    # out = out.transpose((2, 0, 1))
    return out

def read_packedraw(rawPath, width, height, stride):
    # rawPath : path of the raw file
    # width,height: size of the real data
    # stride: linestep of the raw data
    # define output
    raw16bit = np.zeros([height, width])
    file = open(rawPath, 'rb')
    rawdata = file.read()
    file.close()

    # print(sys.getsizeof(rawdata))
    # calc the stride
    # if 0 == ((width * 5 / 4) % 16):
    #     stride = (width * 5 / 4)
    # else:
    #     stride = (width * 5 / 4) - ((width * 5 / 4) % 16) + 16

    stride = (width * 5 / 4)

    rawdata = Image.frombytes('L', (int(stride), int(height)), rawdata)
    rawdata = np.asarray(rawdata)

    data_per_line = int(width * 5 / 4)
    # data_per_line = int(stride)
    first_byte = np.uint16(rawdata[:, 0:data_per_line:5])
    second_byte = np.uint16(rawdata[:, 1:data_per_line:5])
    third_byte = np.uint16(rawdata[:, 2:data_per_line:5])
    fourth_byte = np.uint16(rawdata[:, 3:data_per_line:5])
    fifth_byte = np.uint16(rawdata[:, 4:data_per_line:5])

    # get 10bit data
    first_pixel = np.bitwise_or(np.left_shift(first_byte, 2), np.bitwise_and(fifth_byte, 3))
    second_pixel = np.bitwise_or(np.left_shift(second_byte, 2), np.right_shift(np.bitwise_and(fifth_byte, 12), 2))
    third_pixel = np.bitwise_or(np.left_shift(third_byte, 2), np.right_shift(np.bitwise_and(fifth_byte, 48), 4))
    fourth_pixel = np.bitwise_or(np.left_shift(fourth_byte, 2), np.right_shift(np.bitwise_and(fifth_byte, 192), 6))

    raw16bit[:, 0:width:4] = first_pixel
    raw16bit[:, 1:width:4] = second_pixel
    raw16bit[:, 2:width:4] = third_pixel
    raw16bit[:, 3:width:4] = fourth_pixel

    # Show Image ONLY For debug
    # raw16bit = Image.fromarray(raw16bit)
    # raw16bit.show()
    return raw16bit


def Convert_Unpacked10bit(raw16bit, width, height):
    raw16bit = np.uint16(raw16bit)
    first_pixel = raw16bit[:, 0:width:4]
    second_pixel = raw16bit[:, 1:width:4]
    third_pixel = raw16bit[:, 2:width:4]
    fourth_pixel = raw16bit[:, 3:width:4]
    first_byte = np.right_shift(first_pixel, 2)
    second_byte = np.right_shift(second_pixel, 2)
    third_byte = np.right_shift(third_pixel, 2)
    fourth_byte = np.right_shift(fourth_pixel, 2)
    first_two_bit = np.bitwise_and(first_pixel, 3)
    second_two_bit = np.bitwise_and(second_pixel, 3)
    third_two_bit = np.bitwise_and(third_pixel, 3)
    fourth_two_bit = np.bitwise_and(fourth_pixel, 3)
    fifth_byte = first_two_bit + np.left_shift(second_two_bit, 2) + np.left_shift(third_two_bit, 4) + np.left_shift(
        fourth_two_bit, 6)
    data_per_line = int(width * 5 / 4)
    MIPI_10BIT = np.zeros([height, data_per_line])
    MIPI_10BIT[:, 0:data_per_line:5] = np.uint8(first_byte)
    MIPI_10BIT[:, 1:data_per_line:5] = np.uint8(second_byte)
    MIPI_10BIT[:, 2:data_per_line:5] = np.uint8(third_byte)
    MIPI_10BIT[:, 3:data_per_line:5] = np.uint8(fourth_byte)
    MIPI_10BIT[:, 4:data_per_line:5] = np.uint8(fifth_byte)

    return MIPI_10BIT
