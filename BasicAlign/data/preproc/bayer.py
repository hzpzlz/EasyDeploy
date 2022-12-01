#!/usr/bin python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/23
# @Author  : jishilong@xiaomi.com

# This module improves upon https://arxiv.org/pdf/1904.12945.pdf with code in
# https://github.com/Jiaming-Liu/BayerUnifyAug

import numpy as np
from .base import symmetry


def _bayer_unify_offset(input_pattern, target_pattern):
    """
    This private function return offset from input_pattern to target_pattern.
    Args:
        input_pattern:
        target_pattern:

    Returns:

    """
    if input_pattern == target_pattern:
        offset = [0, 0]
    elif input_pattern[0] == target_pattern[2] and input_pattern[1] == target_pattern[3]:
        offset = [1, 0]
    elif input_pattern[0] == target_pattern[1] and input_pattern[2] == target_pattern[3]:
        offset = [0, 1]
    elif input_pattern[0] == target_pattern[3] and input_pattern[1] == target_pattern[2]:
        offset = [1, 1]
    else:
        raise RuntimeError('Unexpected pair of input and target bayer pattern!')

    return offset


def _bayer_symmetry_converter(bayer_pattern, flip_ud, flip_lr, transpose, reverse=False):
    """
    This private function convert bayer_pattern to the one after 3 common augment ops including flip along height
    flip along width and transpose.
    Args:
        flip_ud:
        flip_lr:
        transpose:
        bayer_pattern: input bayer pattern
        reverse: whether to reverse the order of operations

    Returns:
        aug_pattern: output bayer pattern after augment ops
    """
    if not reverse:
        if flip_ud:
            bayer_pattern = bayer_pattern[2] + bayer_pattern[3] + bayer_pattern[0] + bayer_pattern[1]
        if flip_lr:
            bayer_pattern = bayer_pattern[1] + bayer_pattern[0] + bayer_pattern[3] + bayer_pattern[2]
        if transpose:
            bayer_pattern = bayer_pattern[0] + bayer_pattern[2] + bayer_pattern[1] + bayer_pattern[3]
    else:
        if transpose:
            bayer_pattern = bayer_pattern[0] + bayer_pattern[2] + bayer_pattern[1] + bayer_pattern[3]
        if flip_lr:
            bayer_pattern = bayer_pattern[1] + bayer_pattern[0] + bayer_pattern[3] + bayer_pattern[2]
        if flip_ud:
            bayer_pattern = bayer_pattern[2] + bayer_pattern[3] + bayer_pattern[0] + bayer_pattern[1]
    return bayer_pattern


def bayer_unify(raw, input_pattern, target_pattern, mode):
    """
    Convert bayer raw images from one bayer pattern to another.
    Parameters
    ----------
    raw : np.ndarray in shape (h, w, ...)
        Bayer raw images to be unified.
    input_pattern : {"rggb", "bggr", "grbg", "gbrg"}
        The bayer pattern of the input image.
    target_pattern : {"rggb", "bggr", "grbg", "gbrg"}
        The expected output pattern.
    mode: {"crop", "pad"}
        The way to handle submosaic shift. "crop" abandons the outmost pixels,
        and "pad" introduces extra pixels. Use "crop" in training and "pad" in
        testing.
    """
    offset = _bayer_unify_offset(input_pattern, target_pattern)

    if mode == "pad":
        pad_width = [[offset[0], offset[0]], [offset[1], offset[1]]] + [[0, 0]] * (raw.ndim - 2)
        out = np.pad(raw, pad_width, 'reflect')
    elif mode == "crop":
        h, w = raw.shape[:2]
        out = raw[offset[0]:h - offset[0], offset[1]:w - offset[1]]
    else:
        raise ValueError('Unknown normalization mode!')

    return out


def bayer_symmetry(raw, bayer_pattern, flip_ud, flip_lr, transpose):
    """
    Apply symmetry augmentation to bayer raw images, regardless of whether the shape changes.
    Parameters
    ----------
    raw : np.ndarray in shape (h, w, ...)
        Bayer raw image to be augmented. h and w must be even numbers.
    flip_ud : bool
        If True, do vertical flip.
    flip_lr : bool
        If True, do horizontal flip.
    transpose : bool
        If True, do transpose.
    bayer_pattern : {"rggb", "bggr", "grbg", "gbrg"}
        The bayer pattern of the input images.
    """

    if raw.shape[0] % 2 == 1 or raw.shape[1] % 2 == 1:
        raise ValueError('raw should have even number of height and width!')

    raw = symmetry(raw, flip_ud, flip_lr, transpose)
    aug_pattern = _bayer_symmetry_converter(bayer_pattern, flip_ud, flip_lr, transpose)
    return bayer_unify(raw, aug_pattern, bayer_pattern, "crop")


def bayer_random_crop(raw, input_pattern, target_pattern, crop_height, crop_width):
    """
    Random crop raw image with input_pattern to target_pattern.
    Args:
        raw: np.ndarray in shape (h, w, ...)
        input_pattern:
        target_pattern:
        crop_height:
        crop_width:

    Returns:

    """
    h, w = raw.shape[:2]
    if h % 2 == 1 or w % 2 == 1:
        raise ValueError('raw should have even number of height and width!')
    if crop_height % 2 == 1 or crop_width % 2 == 1:
        raise ValueError('patch should have even number of height and width!')
    assert crop_height <= h and crop_width <= w, 'WRONG RANDOM CROP SHAPE!'

    offset = _bayer_unify_offset(input_pattern, target_pattern)
    hh = np.random.randint(h//2 - crop_height//2 + 1 - offset[0]) * 2 + offset[0]
    ww = np.random.randint(w//2 - crop_width//2 + 1 - offset[1]) * 2 + offset[1]

    return raw[hh:hh + crop_height, ww:ww + crop_width]


def bayer_symmetry_random_crop(raw, input_pattern, target_pattern, flip_ud, flip_lr, transpose,
                               crop_height, crop_width):
    """
    We implement this function for effectiveness. It is just composite function of bayer_symmetry and bayer_random_crop.
    Args:
        raw:
        input_pattern:
        target_pattern:
        flip_ud:
        flip_lr:
        transpose:
        crop_height:
        crop_width:

    Returns:

    """
    h, w = raw.shape[:2]
    if h % 2 == 1 or w % 2 == 1:
        raise ValueError('raw should have even number of height and width!')
    if crop_height % 2 == 1 or crop_width % 2 == 1:
        raise ValueError('patch should have even number of height and width!')
    assert crop_height <= h and crop_width <= w, 'WRONG RANDOM CROP SHAPE!'
    # aug_pattern convert to target_pattern after symmetry
    aug_pattern = _bayer_symmetry_converter(target_pattern, flip_ud, flip_lr, transpose, reverse=True)
    # get offset from input_pattern to aug_pattern
    offset = _bayer_unify_offset(input_pattern, aug_pattern)
    hh = np.random.randint(h//2 - crop_height//2 + 1 - offset[0]) * 2 + offset[0]
    ww = np.random.randint(w//2 - crop_width//2 + 1 - offset[1]) * 2 + offset[1]

    patch = raw[hh:hh + crop_height, ww:ww + crop_width]
    return symmetry(patch, flip_ud, flip_lr, transpose)

