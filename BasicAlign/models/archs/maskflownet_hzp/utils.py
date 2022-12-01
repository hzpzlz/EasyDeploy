import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import ops

import sys

def dila_conv(ch_in, ch_out, kernel_size=3, stride=1, dilation=1, bias=True):
    return nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, dilation=dilation,
              padding=((kernel_size - 1) * dilation) // 2, bias=bias)

def deformable_conv(in_planes, out_planes, kernel_size=3, strides=1, padding=1, use_bias=True):
    return ops.DeformConv2d(in_planes, out_planes, kernel_size, strides, padding, bias=use_bias)

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

class FlowAndMaskEstimator(nn.Module):
    def __init__(self, ch_in, ch_feat):
        super(FlowAndMaskEstimator, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 96, 3, 1, 1)
        self.conv4 = nn.Conv2d(96, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)

        self.relu = nn.LeakyReLU(0.1,inplace=True)

        self.upfeat = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=True)
        self.trade = nn.Conv2d(in_channels = 16, out_channels = ch_feat, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
        self.flow = nn.Conv2d(in_channels = 32+2, out_channels = 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
        self.mask = nn.Conv2d(in_channels = 32+1, out_channels = 1, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)

    #def forward(self, x1, upfeat, corr, flow, mask):
    def forward(self, x1, corr, flow, mask):
        x = torch.cat([x1, corr, flow, mask], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        upfeat = self.relu(self.upfeat(x))
        tradeoff = self.trade(upfeat)
        flow = self.flow(torch.cat([x, flow], dim=1))
        mask = self.mask(torch.cat([x, mask], dim=1))

        return tradeoff, upfeat, flow, mask

class FlowAndMaskEstimator_v2(nn.Module):
    def __init__(self, ch_in, ch_feat):
        super(FlowAndMaskEstimator_v2, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 96, 3, 1, 1)
        self.conv4 = nn.Conv2d(96, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)

        self.relu = nn.LeakyReLU(0.1,inplace=True)

        self.upfeat = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=True)
        self.trade = nn.Conv2d(in_channels = 16, out_channels = ch_feat, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
        self.flow = nn.Conv2d(in_channels = 32+2, out_channels = 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
        self.mask = nn.Conv2d(in_channels = 32+1, out_channels = 1, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
    
    def forward(self, x1, upfeat, corr, flow, mask):
        x = torch.cat([x1, upfeat, corr, flow, mask], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        upfeat = self.relu(self.upfeat(x))
        tradeoff = self.trade(upfeat)
        flow = self.flow(torch.cat([x, flow], dim=1))
        mask = self.mask(torch.cat([x, mask], dim=1))

        return tradeoff, upfeat, flow, mask


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

class DeformableNet(nn.Module):
    def __init__(self, ch_in):
        super(DeformableNet, self).__init__()

        self.deform_conv = deformable_conv(ch_in, ch_in)

    def forward(self, a, b):
        out = self.deform_conv(a, b)

        return out

class ConvNet(nn.Module):
    def __init__(self, ch_in, c=18):
        super(ConvNet, self).__init__()

        self.conv = nn.Conv2d(ch_in+c, ch_in, 3, 1, 1)

    def forward(self, a, b):
        out = self.conv(torch.cat([a, b], dim=1))

        return out

###参考GMA，加入attention和global motion  GMA的pytorch版本
class Attention(nn.Module):
    def __init__(self, ch_in, dim_head=128):
        super(Attention, self).__init__()
        self.scale = dim_head ** -0.5
        self.to_qk = nn.Conv2d(ch_in, dim_head*2, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k = self.to_qk(x).chunk(2, dim=1)
        q = q.unsqueeze(1).permute(0, 1, 3, 4, 2)
        k = k.unsqueeze(1).permute(0, 1, 3, 4, 2)
        q = self.scale * q

        sim = torch.matmul(q.view(b, 1, h*w, -1), k.view(b, 1, h*w, -1).transpose(2, 3)).reshape(b, 1, h, w, h, w)
        sim = sim.reshape(b, 1, h*w, h*w)
        attn = sim.softmax(dim=-1)

        return attn

class Aggregate(nn.Module):
    def __init__(self, ch_in, dim_head=128):
        super(Aggregate, self).__init__()
        self.scale = dim_head ** -0.5
        self.to_v = nn.Conv2d(ch_in, dim_head, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

        if ch_in != dim_head:
            self.project = nn.Conv2d(dim_head, ch_in, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        b, c, h, w = fmap.shape
        v = self.to_v(fmap).unsqueeze(1).permute(0,1,3,4,2).view(b, 1, h*w, -1)
        out = torch.matmul(attn, v).view(b,1,h*w, -1).permute(0, 1, 3,2).reshape(b, -1, h, w)

        if self.project is not None:
            out = self.project(out)
        out = fmap + self.gamma * out
        return out


class AttAggFME(nn.Module):
    def __init__(self, ch_in, in_feat, out_feat, corr_dim=81, mf_dim=128, flow_dim=2, mask_dim=1):
        super(AttAggFME, self).__init__()
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.net_dim = ch_in//2
        self.inp_dim = ch_in - self.net_dim
        self.att = Attention(self.inp_dim)

        self.convc1 = nn.Conv2d(corr_dim, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv_motion = nn.Conv2d(64 + 192, mf_dim - 2, 3, padding=1)

        self.agg = Aggregate(mf_dim)

        self.conv1 = dila_conv(self.net_dim+in_feat+mf_dim*2, 128, 3, 1, 1)
        self.conv2 = dila_conv(128, 128, 3, 1, 2)
        self.conv2 = dila_conv(128, 96, 3, 1, 4)
        self.conv3 = dila_conv(96, 64, 3, 1, 8)
        self.conv4 = dila_conv(64, 32, 3, 1, 1)

        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.upfeat = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=True)
        self.trade = nn.Conv2d(in_channels=16, out_channels=out_feat, kernel_size=3, stride=1, padding=1, dilation=1,
                               groups=1, bias=True)
        self.flow = nn.Conv2d(in_channels=32, out_channels=flow_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                              groups=1, bias=True)
        self.mask = nn.Conv2d(in_channels=32, out_channels=mask_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                              groups=1, bias=True)

    def forward(self, x1, upfeat, corr, flow):
        net, inp = torch.split(x1, self.net_dim, dim=1)
        attention = self.att(inp)
        cor = self.relu(self.convc1(corr))
        cor = self.relu(self.convc2(cor))
        flo = self.relu(self.convf1(flow))
        flo = self.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        motion_fea = torch.cat([self.relu(self.conv_motion(cor_flo)), flow], dim=1)

        motion_fea_global = self.agg(attention, motion_fea)

        x = torch.cat([net, upfeat, motion_fea, motion_fea_global], dim=1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        upfeat = self.relu(self.upfeat(x))
        tradeoff = self.trade(upfeat)
        flow = self.flow(x)
        mask = self.mask(x)

        return tradeoff, upfeat, flow, mask  # x1对应的尺寸 16 2 1

def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([torchHorizontal, torchVertical], 1)

    return grid

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
