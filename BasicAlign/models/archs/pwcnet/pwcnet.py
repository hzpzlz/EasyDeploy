import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .modules import WarpingLayer, FeatureExtractor, CostVolumeLayer, ContextNetwork, FlowEstimator, warp, Correlation_layer, corr_hzp
from ..correlation_package.correlation import Correlation


class Net(nn.Module):
    def __init__(self, output_level=4):
        super(Net, self).__init__()
        lv_chs = [3, 16, 32, 64, 96, 128, 196]
        search_range = 4
        self.residual = True
        self.num_levels = len(lv_chs)
        self.output_level = output_level
        #lv_chs = [3, 16, 32, 64, 96, 128, 196]

        self.relu = nn.LeakyReLU(0.1)
        #self.warping_layer = WarpingLayer()

        self.lv1 = FeatureExtractor(3, 16)
        self.lv2 = FeatureExtractor(16, 32)
        self.lv3 = FeatureExtractor(32, 64)
        self.lv4 = FeatureExtractor(64, 96)
        self.lv5 = FeatureExtractor(96, 128)
        self.lv6 = FeatureExtractor(128, 196)

        #self.corr = Correlation(pad_size=search_range, kernel_size=1, max_displacement=search_range, stride1=1, stride2=1, corr_multiply=1)
        #self.corr = CostVolumeLayer()
        #self.corr = Correlation_layer()
        self.corr = corr_hzp()

        corr_dim = (search_range * 2 + 1) ** 2  # 9*9

        #chs = corr_dim + 196 + 16 + 2 + 1
        chs = corr_dim + 196 + 2
        self.fms6 = FlowEstimator(chs)

        #chs = corr_dim + 128 + 16 + 2 + 1
        chs = corr_dim + 128 + 2
        self.fms5 = FlowEstimator(chs)

        #chs = corr_dim + 96 + 16 + 2 + 1
        chs = corr_dim + 96 + 2
        self.fms4 = FlowEstimator(chs)

        #chs = corr_dim + 64 + 16 + 2 + 1
        chs = corr_dim + 64 + 2
        self.fms3 = FlowEstimator(chs)

        #chs = corr_dim + 32 + 16 + 2 + 1
        chs = corr_dim + 32 + 2
        self.fms2 = FlowEstimator(chs)

        #c_chs = 196 + 16 + 2 + 1
        c_chs = 196 + 2
        self.cnet6 = ContextNetwork(c_chs)

        #c_chs = 128 + 16 + 2 + 1
        c_chs = 128 + 2
        self.cnet5 = ContextNetwork(c_chs)

        #c_chs = 96 + 16 + 2 + 1
        c_chs = 96 + 2
        self.cnet4 = ContextNetwork(c_chs)

        #c_chs = 64 + 16 + 2 + 1
        c_chs = 64 + 2
        self.cnet3 = ContextNetwork(c_chs)

        #c_chs = 32 + 16 + 2 + 1
        c_chs = 32 + 2
        self.cnet2 = ContextNetwork(c_chs)

        #d_chs = 128
        #self.deform5 = DeformableNet(d_chs)
        #self.deform5 = ConvNet(d_chs)

        #d_chs = 96
        #.deform4 = DeformableNet(d_chs)
        #self.deform4 = ConvNet(d_chs)

        #d_chs = 64
        #self.deform3 = DeformableNet(d_chs)
        #self.deform3 = ConvNet(d_chs)

        #d_chs = 32
        #self.deform2 = DeformableNet(d_chs)
        #self.deform2 = ConvNet(d_chs)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x1_raw, x2_raw, test_mode=False):
        x1_lv1 = self.lv1(x1_raw)
        x2_lv1 = self.lv1(x2_raw)

        x1_lv2 = self.lv2(x1_lv1)
        x2_lv2 = self.lv2(x2_lv1)

        x1_lv3 = self.lv3(x1_lv2)
        x2_lv3 = self.lv3(x2_lv2)

        x1_lv4 = self.lv4(x1_lv3)
        x2_lv4 = self.lv4(x2_lv3)

        x1_lv5 = self.lv5(x1_lv4)
        x2_lv5 = self.lv5(x2_lv4)

        x1_lv6 = self.lv6(x1_lv5)
        x2_lv6 = self.lv6(x2_lv5)

        #shape = list(x1_lv6.size());shape[1] = 2
        tb, tc, th, tw = tuple(int(i) for i in  x1_lv6.shape)
        shape = [tb, 2, th, tw]
        flow6 = torch.zeros(shape).to(x1_lv6.device)
        shape[1] = 1
        mask6 = torch.ones(shape).to(x1_lv6.device)
        #upfeat6 = torch.zeros(x1.size()).to(x1.device)

        warp6 = warp(x2_lv6, flow6)

        corr6 = self.relu(self.corr(x1_lv6, warp6))
        #tradeoff5, upfeat5, flow_coarse6, mask6 = self.fms6(x1_lv6, upfeat6, corr6, flow6, mask6)
        flow_coarse6 = self.fms6(x1_lv6, corr6, flow6)
        if self.residual:
            flow_coarse6 = flow_coarse6 + flow6
        #flow_fine6 = self.cnet6(torch.cat([x1_lv6, upfeat6, flow6, mask6], dim=1))
        flow_fine6 = self.cnet6(torch.cat([x1_lv6, flow6], dim=1))
        flow6 = flow_coarse6 + flow_fine6

        flow5 = F.interpolate(flow6, scale_factor = 2, mode = 'bilinear', align_corners=True) * 2

        warp5 = warp(x2_lv5, flow5)

        corr5 = self.relu(self.corr(x1_lv5, warp5))
        #tradeoff4, upfeat4, flow_coarse5, mask5 = self.fms5(x1_lv5, upfeat5, corr5, flow5, mask5)
        flow_coarse5 = self.fms5(x1_lv5, corr5, flow5)
        if self.residual:
            flow_coarse5 = flow_coarse5 + flow5
        #flow_fine5 = self.cnet5(torch.cat([x1_lv5, upfeat5, flow5, mask5], dim=1))
        flow_fine5 = self.cnet5(torch.cat([x1_lv5, flow5], dim=1))
        flow5 = flow_coarse5 + flow_fine5

        flow4 = F.interpolate(flow5, scale_factor=2, mode='bilinear', align_corners=True) * 2

        warp4 = warp(x2_lv4, flow4)

        corr4 = self.relu(self.corr(x1_lv4, warp4))
        #tradeoff3, upfeat3, flow_coarse4, mask4 = self.fms4(x1_lv4, upfeat4, corr4, flow4, mask4)
        flow_coarse4 = self.fms4(x1_lv4, corr4, flow4)
        if self.residual:
            flow_coarse4 = flow_coarse4 + flow4
        #flow_fine4 = self.cnet4(torch.cat([x1_lv4, upfeat4, flow4, mask4], dim=1))
        flow_fine4 = self.cnet4(torch.cat([x1_lv4, flow4], dim=1))
        flow4 = flow_coarse4 + flow_fine4

        flow3 = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=True) * 2

        warp3 = warp(x2_lv3, flow3)

        corr3 = self.relu(self.corr(x1_lv3, warp3))
        #tradeoff2, upfeat2, flow_coarse3, mask3 = self.fms3(x1_lv3, upfeat3, corr3, flow3, mask3)
        flow_coarse3 = self.fms3(x1_lv3, corr3, flow3)
        if self.residual:
            flow_coarse3 = flow_coarse3 + flow3
        #flow_fine3 = self.cnet3(torch.cat([x1_lv3, upfeat3, flow3, mask3], dim=1))
        flow_fine3 = self.cnet3(torch.cat([x1_lv3, flow3], dim=1))
        flow3 = flow_coarse3 + flow_fine3

        flow2 = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=True) * 2

        warp2 = warp(x2_lv2, flow2)

        corr2 = self.relu(self.corr(x1_lv2, warp2))
        #_, _, flow_coarse2, mask2 = self.fms2(x1_lv2, upfeat2, corr2, flow2, mask2)
        flow_coarse2 = self.fms2(x1_lv2, corr2, flow2)
        if self.residual:
            flow_coarse2 = flow_coarse2 + flow2
        #flow_fine2 = self.cnet2(torch.cat([x1_lv2, upfeat2, flow2, mask2], dim=1))
        flow_fine2 = self.cnet2(torch.cat([x1_lv2, flow2], dim=1))
        flow2 = flow_coarse2 + flow_fine2

        flow2 = F.interpolate(flow2, scale_factor=2 ** (self.num_levels - self.output_level - 1), mode='bilinear',
                              align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)

        if test_mode == False:
            return [flow6, flow5, flow4, flow3, flow2]
        else:
            return flow2
