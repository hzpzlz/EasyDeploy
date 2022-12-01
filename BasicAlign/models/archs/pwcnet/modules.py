import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def warp(img, flow, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    #batch, dim, H, W = img.shape
    batch, dim, H, W = tuple(int(i) for i in img.shape)
    basic_grid = coords_grid(batch, H, W).cuda()
    coords = basic_grid + flow 
    #xgrid, ygrid = coords.permute(0, 2, 3, 1).split([1,1], dim=-1)
    #coords_ = coords.permute(0, 2, 3, 1)
    #xgrid, ygrid = coords_.split([1,1], dim=-1)
    xgrid, ygrid = torch.split(coords, [1,1], dim=1)

    xgrid = xgrid.permute(0,2,3,1).reshape(-1)
    ygrid = ygrid.permute(0,2,3,1).reshape(-1)

    out_size = [H, W]

    #img = img.permute([0, 2, 3, 1])
    wo = _interpolate(img, xgrid, ygrid, out_size)
    #print(wo.shape, "nnnnnnnnnnnnnnnnnnn")
    wo = wo.reshape(-1, out_size[0], out_size[1], dim).permute(0, 3, 1, 2)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return wo

def _interpolate(im, x, y, out_size):
    #num_batch = im.size(0)
    #channels = im.size(1)
    #height = im.size(2)
    #width = im.size(3)
    num_batch, channels, height, width = tuple(int(i) for i in im.shape)

    x = x.float()
    y = y.float()

    out_height = out_size[0]
    out_width = out_size[1]
    zero = 0
    max_y = int(height - 1)
    max_x = int(width - 1)
    #max_y = height - 1
    #max_x = width - 1

    #x0 = x.floor().int()
    #y0 = y.floor().int()
    #x1 = x0 + 1
    #y1 = y0 + 1
    x0 = x.floor()
    y0 = y.floor()
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = torch.clamp(x0, zero, max_x).type(torch.cuda.LongTensor)
    x1 = torch.clamp(x1, zero, max_x).type(torch.cuda.LongTensor)
    y0 = torch.clamp(y0, zero, max_y).type(torch.cuda.LongTensor)
    y1 = torch.clamp(y1, zero, max_y).type(torch.cuda.LongTensor)
    dim2 = width
    dim1 = width * height

    base = _repeat(torch.arange(0, num_batch).int() * dim1, out_height * out_width)
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    im = im.permute(0, 2, 3, 1)
    im_flat = im.reshape(-1, channels).float()

    # print("interpolating... w/")
    # input(idx_a)

    #Ia = torch.index_select(im_flat, dim=0,
    #                        index=Variable(idx_a.type(torch.cuda.LongTensor)))  # .type(torch.cuda(3).FloatTensor)
    #Ib = torch.index_select(im_flat, dim=0,
    #                        index=Variable(idx_b.type(torch.cuda.LongTensor)))  # .type(torch.cuda(3).FloatTensor)
    #Ic = torch.index_select(im_flat, dim=0,
    #                        index=Variable(idx_c.type(torch.cuda.LongTensor)))  # .type(torch.cuda(3).FloatTensor)
    #Id = torch.index_select(im_flat, dim=0,
    #                        index=Variable(idx_d.type(torch.cuda.LongTensor)))  # .type(torch.cuda(3).FloatTensor)
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()
    wa = ((x1_f - x) * (y1_f - y))[:, None]
    wb = ((x1_f - x) * (y - y0_f))[:, None]
    wc = ((x - x0_f) * (y1_f - y))[:, None]
    wd = ((x - x0_f) * (y - y0_f))[:, None]

    A = wa * Ia.cuda().type(torch.cuda.FloatTensor)
    B = wb * Ib.cuda().type(torch.cuda.FloatTensor)
    C = wc * Ic.cuda().type(torch.cuda.FloatTensor)
    D = wd * Id.cuda().type(torch.cuda.FloatTensor)

    output = A + B + C + D
    return output

def _repeat(x, n_repeats):
    rep = torch.ones([1, n_repeats]).int()
    # There's some differnent between my implementation and original'
    # If something wrong, should change it back to original type
    x = torch.matmul(x.view(-1, 1), rep).cuda()
    return x.view(-1)

class CostVolumeLayer(nn.Module):
    def __init__(self, search_range=4):
        super(CostVolumeLayer, self).__init__()
        self.search_range = search_range

    def forward(self, x1, x2):
        search_range = self.search_range
        #shape = list(x1.size()); shape[1] = (self.search_range * 2 + 1) ** 2
        b,c,h,w = tuple(int(i) for i in x1.shape)#; shape[1] = (self.search_range * 2 + 1) ** 2
        shape = [b, (self.search_range * 2 + 1) ** 2, h, w]
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

class Correlation_layer(nn.Module): 
    def __init__(self, search_range=4): 
        super(Correlation_layer, self).__init__() 
        self.search_range = search_range

    def forward(self, feature1, feature2):
        feature2 = F.pad(feature2, (self.search_range, ) * 4, mode='constant', value=0)
        #feature2 = F.pad(feature2, (self.search_range, ) * 4, mode='replicate')
        #_,_,h,w = feature1.shape
        _, _, h, w = tuple(int(i) for i in feature1.shape)

        dot = []

        for idx in range(2 * self.search_range + 1):
            for jdx in range(2 * self.search_range + 1):
                dot.append(torch.sum(feature1 * feature2[:,:,idx:idx+h,jdx:jdx+w], dim=1, keepdim=True))

        return torch.cat(dot,1)

def make_mesh(patch_w,patch_h):

    x_flat = torch.arange(0, patch_w)
    x_flat = x_flat.unsqueeze(0)
    y_one = torch.ones(patch_h)
    y_one = y_one.unsqueeze(-1)
    x_mesh=torch.matmul(y_one, x_flat.float())

    y_flat = torch.arange(0, patch_h)
    y_flat = y_flat.unsqueeze(-1)
    x_one = torch.ones(patch_w)
    x_one = x_one.unsqueeze(0)
    y_mesh = torch.matmul(y_flat.float(), x_one)

    return x_mesh.cuda(), y_mesh.cuda()

def patch_slice(fea, x_mesh, y_mesh, bt, cn, h, w, ph, pw, x, y):
    batch_indexs = torch.arange(0, bt * cn * h * w, cn * h * w).unsqueeze(1).expand(bt, cn * ph * pw).cuda()
    #x_mesh, y_mesh = make_mesh(pw, ph)
    y_t_flat = y_mesh.reshape(-1)
    x_t_flat = x_mesh.reshape(-1)

    patch_indices = (y_t_flat + y) * w + (x_t_flat + x)

    patch_indices = patch_indices.reshape(1, -1).repeat(cn, 1)
    bb = torch.arange(0, cn) * h * w

    patch_inx = patch_indices + bb.reshape(cn, 1).cuda()

    indexs = batch_indexs.long() + patch_inx.reshape(1, -1).long()
    patch_img = torch.gather(fea.reshape(-1), 0, indexs.reshape(-1))
    out = patch_img.reshape(bt, cn, ph, pw)

    return out

class corr_hzp(nn.Module):
    def __init__(self, k=4):
        super(corr_hzp, self).__init__()
        self.k = k
    def forward(self, x, y):
        b, c, ph, pw = tuple(int(i) for i in y.shape)
        #x_mesh, y_mesh = make_mesh(pw, ph)
        k = self.k
        y = F.pad(y, (k,k,k,k), 'reflect')
        #print(y.shape, ph, pw, "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
        h, w = tuple(int(i) for i in y.shape[-2:])

        result = []
        for i in np.arange(-k, k+1, 1):
            for j in np.arange(-k, k+1, 1):
                sx = k+i
                sy = k+j
                #corr = torch.cosine_similarity(x, y[:, :, sx:sx+h, sy:sy+w], dim=1)
                corr = torch.sum(x*y[:, :, sx:sx + ph, sy:sy + pw], dim=1)
                #corr = torch.sum(x*patch_slice(y, x_mesh, y_mesh, b, c, h, w, ph, pw, sx, sy), dim=1)
                result.append(corr)

        out = torch.stack(result, dim=1)
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, 2, 1)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, 1, 1)
        self.conv3 = nn.Conv2d(ch_out, ch_out, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        return x

class FlowEstimator(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimator, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 96, 3, 1, 1)
        self.conv4 = nn.Conv2d(96, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)

        self.relu = nn.LeakyReLU(0.1,inplace=True)

        self.flow = nn.Conv2d(in_channels = 32, out_channels = 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)

    #def forward(self, x1, upfeat, corr, flow, mask):
    def forward(self, x1, corr, flow):
        x = torch.cat([x1, corr, flow], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        flow = self.flow(x)

        return flow

def dila_conv(ch_in, ch_out, kernel_size=3, stride=1, dilation=1, bias=True):
    return nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, dilation=dilation,
              padding=((kernel_size - 1) * dilation) // 2, bias=bias)

class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.conv1 = dila_conv(ch_in, 128, 3, 1, 1)
        self.conv2 = dila_conv(128, 128, 3, 1, 2)
        self.conv3 = dila_conv(128, 128, 3, 1, 4)
        self.conv4 = dila_conv(128, 96, 3, 1, 8)
        self.conv5 = dila_conv(96, 64, 3, 1, 16)
        self.conv6 = dila_conv(64, 32, 3, 1, 1)

        self.conv7 = dila_conv(32, 2, 3, 1, 1)

        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))

        flow_fine = self.conv7(x)
        return flow_fine
