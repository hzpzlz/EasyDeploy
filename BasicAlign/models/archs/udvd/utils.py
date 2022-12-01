import torch
from torch.distributions import Poisson
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def get_noise(data, dist='G', noise_std = float(25)/255.0, mode='S',
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    if(dist == 'G'):
        noise_std /= 255.
        min_noise /= 255.
        max_noise /= 255.
        noise = torch.randn_like(data);
        if mode == 'B':
            n = noise.shape[0];
            noise_tensor_array = (max_noise - min_noise) * torch.rand(n) + min_noise;
            for i in range(n):
                noise.data[i] = noise.data[i] * noise_tensor_array[i];
        else:
            noise.data = noise.data * noise_std;
    elif(dist == 'P'):
        noise = torch.randn_like(data);
        if mode == 'S':
            noise_std /= 255.
            noise = torch.poisson(data*noise_std)/noise_std - data
    return noise

def ssim(clean, noisy, normalized=True, raw=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.] else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean = clean.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    noisy = noisy.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)

    if raw:
        noisy = (np.uint16(noisy*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240)

    if normalized:
        return np.array([structural_similarity(c, n, data_range=255, multichannel=True) for c, n in zip(clean, noisy)]).mean()
    else:
        return np.array([structural_similarity(c, n, data_range=1.0, multichannel=True) for c, n in zip(clean, noisy)]).mean()


def psnr(clean, noisy, normalized=True, raw=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean = clean.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    noisy = noisy.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)

    if raw:
        noisy = (np.uint16(noisy*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240)

    if normalized:
        return np.array([peak_signal_noise_ratio(c, n, data_range=255) for c, n in zip(clean, noisy)]).mean()
    else:
        return np.array([peak_signal_noise_ratio(c, n, data_range=1.0) for c, n in zip(clean, noisy)]).mean()

