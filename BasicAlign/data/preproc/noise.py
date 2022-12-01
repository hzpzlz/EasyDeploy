#!/usr/bin python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/22
# @Author  : jishilong@xiaomi.com

import numpy as np
from scipy import stats

def gaussian_read(im, sigma):
    """
    Gaussian read noise. Can also be used as gaussian noise independent of signal.
    Args:
        im:
        sigma:

    Returns:

    """
    return np.random.normal(scale=sigma, size=im.shape)

def tl_read(im, lam, sigma):
    """
    Tukey Lambda read noise.
    Args:
        im:
        lam:
        sigma:

    Returns:

    """
    return stats.tukeylambda.rvs(lam, scale=sigma, size=im.shape)

def poisson_shot(im, k):
    """
    Poisson shot noise. Return raw image added noise.
    Args:
        im:
        k: conversion gain with unit DN/e.

    Returns:

    """
    return np.random.poisson(im / k) * k

def gaussian_shot(im, k):
    """
    Gaussian shot noise, asymptotic distribution of Poisson one.
    Args:
        im:
        k: conversion gain with unit DN/e.

    Returns:

    """
    return np.random.normal(scale=(k * im) ** 0.5)

def gaussian_read_shot(im, k, sigma):
    """
    Summation of gaussian_read and gaussian_shot. We implement this function for effectiveness. By the way,
    this noise is known as heteroscedastic gaussian in some paper.
    Args:
        im:
        k: conversion gain with unit DN/e.
        sigma: standard deviation of read noise. It shall be notice that, in most papers, which model variance
        of the noise as Var(N) = aS + b or so on, have different definition in this parameter.

    Returns:

    """
    return np.random.normal(scale=(k * im + sigma**2) ** 0.5)

def gaussian_row(im, sigma):
    """
    Gaussian row noise.
    Args:
        im: unpacked raw image in shape (h, w, ...)
        sigma:

    Returns:

    """
    row_noise_shape = [im.shape[0]] + [1] * (im.ndim - 1)
    return np.random.normal(scale=sigma, size=row_noise_shape)

def uniform_quant(im, black_level, white_level):
    """
    Quantization noise with uniform distribution.
    Args:
        im:
        black_level:
        white_level:

    Returns:

    """
    black_level = np.mean(black_level)
    quant_step = 1 / (white_level - black_level)
    return np.random.uniform(-1/2 * quant_step, 1/2 * quant_step, size=im.shape)

def fake_quant(im, black_level, white_level):
    """
    Fake quantization noise. Return raw image added noise.
    n.b. we don't clip the result to range [0, 1] as this operation is common step.
    Args:
        im:
        black_level:
        white_level:

    Returns:

    """
    black_level = np.mean(black_level)
    im = np.round(im * (white_level - black_level) + black_level)
    return (im - black_level) / (white_level - black_level)

def add_noise(im,params_1600_dict,noise_model_dict,noise_comp):

    if noise_comp[0] == 'G' and noise_comp[1] == 'G':
        im += gaussian_read_shot(im, params_1600_dict['K'], params_1600_dict['sigma_read'])
    else:
        if noise_comp[0] == 'G':
            im += gaussian_shot(im, params_1600_dict['K'])
        elif noise_comp[0] == 'P':
            im = poisson_shot(im, params_1600_dict['K'])

        if noise_comp[1] == 'G':
            im += gaussian_read(im, params_1600_dict['sigma_read'])
        elif noise_comp[1] == 'TL':
            im += tl_read(im, params_1600_dict['lambda'], params_1600_dict['sigma_tl'])

    if noise_comp[2] == 'G':
        im += gaussian_row(im, params_1600_dict['sigma_row'])

    if noise_comp[3] == 'U':
        im += uniform_quant(im, noise_model_dict['black_level'],
                                       noise_model_dict['white_level'])
    elif noise_comp[3] == 'F':
        im = fake_quant(im, noise_model_dict['black_level'],
                                   noise_model_dict['white_level'])

    return im
