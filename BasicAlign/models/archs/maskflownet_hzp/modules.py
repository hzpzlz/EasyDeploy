import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import ops

import sys

def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([torchHorizontal, torchVertical], 1)

    return grid

def conv(batch_norm, in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def deformable_conv(in_planes, out_planes, kernel_size=3, strides=1, padding=1, use_bias=True):
    return ops.DeformConv2d(in_planes, out_planes, kernel_size, strides, padding, bias=use_bias)

class WarpingLayer(nn.Module):
    def __init__(self):
        super(WarpingLayer, self).__init__()
    
    def forward(self, x, flow):
        # WarpingLayer uses F.grid_sample, which expects normalized grid
        # we still output unnormalized flow for the convenience of comparing EPEs with FlowNet2 and original code
        # so here we need to denormalize the flow
        flow_for_grip = torch.zeros_like(flow)
        flow_for_grip[:,0,:,:] = flow[:,0,:,:] / ((flow.size(3) - 1.0) / 2.0)
        flow_for_grip[:,1,:,:] = flow[:,1,:,:] / ((flow.size(2) - 1.0) / 2.0)
        #print(get_grid(x).shape, flow_for_grip.shape, "fffffff")

        grid = (get_grid(x).cuda() + flow_for_grip).permute(0, 2, 3, 1)
        x_warp = F.grid_sample(x, grid, align_corners=True)
        return x_warp


class CostVolumeLayer(nn.Module):

    def __init__(self):
        super(CostVolumeLayer, self).__init__()
        self.search_range = 4

    def forward(self, x1, x2):
        search_range = self.search_range
        shape = list(x1.size()); shape[1] = (self.search_range * 2 + 1) ** 2
        cv = torch.zeros(shape).cuda()

        for i in range(-search_range, search_range + 1):
            for j in range(-search_range, search_range + 1):
                if   i < 0: slice_h, slice_h_r = slice(None, i), slice(-i, None)
                elif i > 0: slice_h, slice_h_r = slice(i, None), slice(None, -i)
                else:       slice_h, slice_h_r = slice(None),    slice(None)

                if   j < 0: slice_w, slice_w_r = slice(None, j), slice(-j, None)
                elif j > 0: slice_w, slice_w_r = slice(j, None), slice(None, -j)
                else:       slice_w, slice_w_r = slice(None),    slice(None)

                cv[:, (search_range*2+1) * i + j, slice_h, slice_w] = (x1[:,:,slice_h, slice_w]  * x2[:,:,slice_h_r, slice_w_r]).sum(1)
    
        return cv / shape[1]

class FeaturePyramidExtractor(nn.Module):
    def __init__(self, lv_chs, batch_norm=False):
        super(FeaturePyramidExtractor, self).__init__()

        #self.convs = []
        self.convs = nn.ModuleList()
        for l, (ch_in, ch_out) in enumerate(zip(lv_chs[:-1], lv_chs[1:])):
            layer = nn.Sequential(
                conv(batch_norm, ch_in, ch_out, stride = 2),
                conv(batch_norm, ch_out, ch_out),
                conv(batch_norm, ch_out, ch_out)
            )
            self.add_module(f'Feature(Lv{l})', layer)
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x); feature_pyramid.append(x)

        return feature_pyramid[::-1]

class OpticalFlowEstimator(nn.Module):
    def __init__(self, ch_in, ch_feat, batch_norm=False):
        super(OpticalFlowEstimator, self).__init__()

        self.convs = nn.Sequential(
            conv(batch_norm, ch_in, 128),
            conv(batch_norm, 128, 128),
            conv(batch_norm, 128, 96),
            conv(batch_norm, 96, 64),
            conv(batch_norm, 64, 32),
        )

        self.relu = nn.LeakyReLU(0.1,inplace=True)

        self.upfeat = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv_f = nn.Conv2d(in_channels = 16, out_channels = ch_feat, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
        self.flow = nn.Conv2d(in_channels = 32, out_channels = 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
        self.mask = nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)

    def forward(self, x):
        base = self.convs(x)

        upfeat = self.conv_f(self.upfeat(base))
        flow = self.flow(base)
        mask = self.mask(base)

        return upfeat, flow, mask

class ContextNetwork(nn.Module):
    def __init__(self, ch_in, batch_norm=False):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(batch_norm, ch_in, 128, 3, 1, 1),
            conv(batch_norm, 128, 128, 3, 1, 2),
            conv(batch_norm, 128, 128, 3, 1, 4),
            conv(batch_norm, 128, 96, 3, 1, 8),
            conv(batch_norm, 96, 64, 3, 1, 16),
            conv(batch_norm, 64, 32, 3, 1, 1),
            conv(batch_norm, 32, 2, 3, 1, 1)
        )
    
    def forward(self, x):
        return self.convs(x)

class DeformableNet(nn.Module):
    def __init__(self, ch_in, batch_norm=False):
        super(DeformableNet, self).__init__()

        self.deform_conv = deformable_conv(ch_in, ch_in)

    def forward(self, a, b):
        out = self.deform_conv(a, b);

        return out
