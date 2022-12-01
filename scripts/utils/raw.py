#!/usr/bin python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/11
# @Author  : jishilong@xiaomi.com

import numpy as np
import math

def bayer_to_offsets(bayer_pattern):
    """
    Transform bayer pattern to offsets in order 'RGrBGb'
    n.b. Support 'RGrBGb' bayer pattern only.
    Args:
        bayer_pattern: string, e.g. 'rggb'. Must be one of 'rggb', 'grbg', 'gbrg', 'bggr'

    Returns:
        offsets: packed raw image with 4 channels
    """
    bayer_pattern = bayer_pattern.lower()
    assert bayer_pattern in ['rggb', 'grbg', 'gbrg', 'bggr'], 'WRONG BAYER PATTERN!'

    if bayer_pattern == 'rggb':
        offsets = [[0,0],[0,1],[1,1],[1,0]]
    elif bayer_pattern == 'grbg':
        offsets = [[0,1],[0,0],[1,0],[1,1]]
    elif bayer_pattern == 'gbrg':
        offsets = [[1,0],[0,0],[0,1],[1,1]]
    else: #bayer_pattern == 'bggr':
        offsets = [[1,1],[0,1],[0,0],[1,0]]

    return offsets

def crop_to_patches(rawim, crop_height, crop_width):
    """
    Crop packed raw image to patches with size (crop_height, crop_width).
    n.b. We only implement central non-overlapped version up to now.
    Args:
        rawim:
        crop_height:
        crop_width:

    Returns:
        patches: ndarray with shape (num_row, num_col, crop_height, crop_width, 4)
    """
    height, width, _ = rawim.shape

    offset_h = (height % crop_height) // 2
    offset_w = (width % crop_width) // 2

    num_r = height // crop_height
    num_c = width // crop_width

    rawim = rawim[offset_h:offset_h + num_r * crop_height,
                  offset_w:offset_w + num_c * crop_width]

    rawim = np.asarray(np.split(rawim, num_c, axis=-2))
    rawim = np.asarray(np.split(rawim, num_r, axis=-3))

    return rawim


# def pack_raw_to_4ch(rawim, bayer_pattern):
#     """
#     Pack raw to h/2 x w/2 x 4n with order "RGrBGb..." RRRRGGGGBBBBGGGG
#     n.b. Support ordinary bayer pattern only.
#     Args:
#         rawim: numpy.ndarray in shape (h, w, ...)
#         bayer_pattern: string, e.g. "rggb". Must be one of "rggb", "grbg", "gbrg", "bggr"
#
#     Returns:
#         out: packed raw image with 4n channels
#     """
#     offsets = bayer_to_offsets(bayer_pattern)
#     if rawim.ndim == 2:
#         rawim = np.expand_dims(rawim, axis=-1)
#     rawim = np.concatenate((rawim[offsets[0][0]::2, offsets[0][1]::2],
#                             rawim[offsets[1][0]::2, offsets[1][1]::2],
#                             rawim[offsets[2][0]::2, offsets[2][1]::2],
#                             rawim[offsets[3][0]::2, offsets[3][1]::2]), axis=-1)
#     return rawim

def pack_raw_to_4ch(rawim, offsets):
    """
    Pack raw to h/2 x w/2 x 4n with order "RGrBGb..." RGBG RGBG RGBG
    n.b. Support ordinary bayer pattern only.
    Args:
        rawim: numpy.ndarray in shape (h, w, ...)
        bayer_pattern: string, e.g. "rggb". Must be one of "rggb", "grbg", "gbrg", "bggr"

    Returns:
        out: packed raw image with 4n channels
    """


    if rawim.ndim == 2:
        rawim = np.expand_dims(rawim, axis=-1)
        rawim_pack = np.concatenate((rawim[offsets[0][0]::2, offsets[0][1]::2],
                                rawim[offsets[1][0]::2, offsets[1][1]::2],
                                rawim[offsets[2][0]::2, offsets[2][1]::2],
                                rawim[offsets[3][0]::2, offsets[3][1]::2]), axis=-1)
    elif rawim.ndim ==3:
        frame_num = rawim.shape[2]
        rawim_pack = np.zeros((int(rawim.shape[0]/2), int(rawim.shape[1]/2), rawim.shape[2] * 4))
        for i in range(frame_num):
            rawim_temp = rawim[:,:,i]
            rawim_temp = np.expand_dims(rawim_temp, axis=-1)
            rawim_temp_pack = np.concatenate((rawim_temp[offsets[0][0]::2, offsets[0][1]::2],
                                              rawim_temp[offsets[1][0]::2, offsets[1][1]::2],
                                              rawim_temp[offsets[2][0]::2, offsets[2][1]::2],
                                              rawim_temp[offsets[3][0]::2, offsets[3][1]::2]), axis=-1)

            rawim_pack[:, :, i * 4:(i + 1) * 4] = rawim_temp_pack

    return rawim_pack

def unpack_raw(rawim, offsets):
    """
    Inverse of pack_raw_to_4ch.
    Args:
        rawim: RGBG TO GRBG
        bayer_pattern:

    Returns:

    """

    h, w, c = rawim.shape
    n = c // 4
    out = np.zeros_like(rawim).reshape((h * 2, w * 2, -1))
    out = np.squeeze(out)

    out[offsets[0][0]::2, offsets[0][1]::2] = np.squeeze(rawim[..., :n])
    out[offsets[1][0]::2, offsets[1][1]::2] = np.squeeze(rawim[..., n:2*n])
    out[offsets[2][0]::2, offsets[2][1]::2] = np.squeeze(rawim[..., 2*n:3*n])
    out[offsets[3][0]::2, offsets[3][1]::2] = np.squeeze(rawim[..., 3*n:])
    return out


def rescaling(rawim, black_level, white_level, clipping=True):
    """
    Try to normalize packed raw image into range [0, 1].
    Args:
        rawim:
        black_level:
        white_level:
        clipping: Boolean. Determine whether clip raw image into range [0, 1].

    Returns:
        rawim: normalized packed raw image with type np.float32
    """
    rawim = rawim.astype(np.float)
    black_level = np.asarray(black_level)
    rawim = (rawim - black_level) / (white_level - black_level)
    if clipping:
        rawim = np.clip(rawim, 0, 1)
    return rawim.astype(np.float32)

# revised by zhangyuqian
# 2020-09-04
def unrescaling(rawim,black_level,white_level):
    """
    Try to scale unpack raw image into range [black_level,white_level]
    :param rawim:
    :param black_level:
    :param white_level:
    :return: rawim with type np.uint16
    """
    rawim = np.minimum(np.maximum(rawim * (white_level - black_level) + black_level, 0), white_level)
    rawim = rawim.astype(np.uint16)
    return rawim

def adjust_rescaling(rawim, ratio, black_level, white_level):
    """
    To make the 0~BL distribution same as BL~WL
    adjust value and normalize packed raw image into range [0, 1].
    Args:
        rawim:
        black_level:
        white_level:
        ratio: ratio to lighten

    Returns:
        rawim: normalized packed raw image with type np.float32
    """
    rawim = rawim.astype(np.float)

    rawim = rawim-black_level
    rawim = np.clip(rawim*ratio,-1*black_level,np.max(rawim*ratio))
    rawim = rawim + black_level
    rawim = np.clip(rawim/white_level,0.0,1.0)

    return rawim.astype(np.float32)

def unrescaling_c(rawim,black_level,white_level):
    """
    Try to scale unpack raw image into range [black_level,white_level]
    :param rawim:
    :param black_level:
    :param white_level:
    :return: rawim with type np.uint16
    """
    black_level = np.asarray(black_level)
    rawim = np.minimum(np.maximum(rawim * (white_level - black_level) + black_level, 0), white_level)
    rawim = rawim.astype(np.uint16)
    return rawim

