#!/usr/bin python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/25
# @Author  : jishilong@xiaomi.com

import numpy as np


def symmetry(im, flip_ud, flip_lr, transpose):
    """
    Return symmetry state of any image with shape (h, w, ...).
    Args:
        im:
        flip_ud:
        flip_lr:
        transpose:

    Returns:

    """
    if flip_ud:
        im = np.flip(im, axis=0)
    if flip_lr:
        im = np.flip(im, axis=1)
    if transpose:
        im = np.moveaxis(im, 0, 1)
    return im

def random_crop(im, crop_height, crop_width):
    """
    Random crop image with shape (h, w, ...) to (crop_height, crop_width, ...).
    Args:
        im:
        crop_height:
        crop_width:

    Returns:

    """
    h, w = im.shape[:2]
    assert crop_height <= h and crop_width <= w, 'WRONG RANDOM CROP SHAPE!'

    hh = np.random.randint(h - crop_height + 1)
    ww = np.random.randint(w - crop_width + 1)
    return im[hh:hh + crop_height, ww:ww + crop_width]
