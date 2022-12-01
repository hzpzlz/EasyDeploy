import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import WarpingLayer, FeatureExtractor, CostVolumeLayer, FlowAndMaskEstimator, ContextNetwork, DeformableNet, ConvNet, FlowAndMaskEstimator_v2
from .utils import AttAggFME
from ..correlation_package.correlation import Correlation


class MaskFlowNet(nn.Module):
    def __init__(self, batch_norm=False):
        super(MaskFlowNet, self).__init__()
        lv_chs = [3, 16, 32, 64, 96, 128, 196]
        search_range = 4
        self.residual = True
        self.num_levels = len(lv_chs)
        self.output_level = 4
        #lv_chs = [3, 16, 32, 64, 96, 128, 196]

        self.relu = nn.LeakyReLU(0.1)
        self.warping_layer = WarpingLayer()

        self.lv1 = FeatureExtractor(3, 16)
        self.lv2 = FeatureExtractor(16, 32)
        self.lv3 = FeatureExtractor(32, 64)
        self.lv4 = FeatureExtractor(64, 96)
        self.lv5 = FeatureExtractor(96, 128)
        self.lv6 = FeatureExtractor(128, 196)

        self.corr = Correlation(pad_size=search_range, kernel_size=1, max_displacement=search_range, stride1=1, stride2=1, corr_multiply=1)

        corr_dim = (search_range * 2 + 1) ** 2  # 9*9

        #chs = corr_dim + 196 + 16 + 2 + 1
        chs = corr_dim + 196 + 2 + 1
        self.fms6 = FlowAndMaskEstimator(chs, 128)

        #chs = corr_dim + 128 + 16 + 2 + 1
        chs = corr_dim + 128 + 2 + 1
        self.fms5 = FlowAndMaskEstimator(chs, 96)

        #chs = corr_dim + 96 + 16 + 2 + 1
        chs = corr_dim + 96 + 2 + 1
        self.fms4 = FlowAndMaskEstimator(chs, 64)

        #chs = corr_dim + 64 + 16 + 2 + 1
        chs = corr_dim + 64 + 2 + 1
        self.fms3 = FlowAndMaskEstimator(chs, 32)

        #chs = corr_dim + 32 + 16 + 2 + 1
        chs = corr_dim + 32 + 2 + 1
        self.fms2 = FlowAndMaskEstimator(chs, 16)

        #c_chs = 196 + 16 + 2 + 1
        c_chs = 196 + 2 + 1
        self.cnet6 = ContextNetwork(c_chs)

        #c_chs = 128 + 16 + 2 + 1
        c_chs = 128 + 2 + 1
        self.cnet5 = ContextNetwork(c_chs)

        #c_chs = 96 + 16 + 2 + 1
        c_chs = 96 + 2 + 1
        self.cnet4 = ContextNetwork(c_chs)

        #c_chs = 64 + 16 + 2 + 1
        c_chs = 64 + 2 + 1
        self.cnet3 = ContextNetwork(c_chs)

        #c_chs = 32 + 16 + 2 + 1
        c_chs = 32 + 2 + 1
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

        shape = list(x1_lv6.size());shape[1] = 2
        flow6 = torch.zeros(shape).to(x1_lv6.device)
        shape[1] = 1
        mask6 = torch.ones(shape).to(x1_lv6.device)
        #upfeat6 = torch.zeros(x1.size()).to(x1.device)

        warp6 = self.warping_layer(x2_lv6, flow6)

        corr6 = self.relu(self.corr(x1_lv6, warp6))
        #tradeoff5, upfeat5, flow_coarse6, mask6 = self.fms6(x1_lv6, upfeat6, corr6, flow6, mask6)
        tradeoff5, _, flow_coarse6, mask6 = self.fms6(x1_lv6, corr6, flow6, mask6)
        if self.residual:
            flow_coarse6 = flow_coarse6 + flow6
        #flow_fine6 = self.cnet6(torch.cat([x1_lv6, upfeat6, flow6, mask6], dim=1))
        flow_fine6 = self.cnet6(torch.cat([x1_lv6, flow6, mask6], dim=1))
        flow6 = flow_coarse6 + flow_fine6

        flow5 = F.interpolate(flow6, scale_factor = 2, mode = 'bilinear', align_corners=True) * 2
        mask5 = F.interpolate(mask6, scale_factor = 2, mode = 'bilinear', align_corners=True)

        # warp5 = flow5.unsqueeze(1)
        # warp5 = torch.repeat_interleave(warp5, 9, 1)
        # S1, S2, S3, S4, S5 = warp5.shape
        # warp5 = warp5.view(S1, S2 * S3, S4, S5)
        # warp5 = self.deform5(x2_lv5, warp5)
        # warp5 = (warp5 /16. * F.sigmoid(mask5)) + tradeoff5
        # warp5 = self.relu(warp5)

        warp5 = self.warping_layer(x2_lv5, flow5) * F.sigmoid(mask5) + tradeoff5

        corr5 = self.relu(self.corr(x1_lv5, warp5))
        #tradeoff4, upfeat4, flow_coarse5, mask5 = self.fms5(x1_lv5, upfeat5, corr5, flow5, mask5)
        tradeoff4, _, flow_coarse5, mask5 = self.fms5(x1_lv5, corr5, flow5, mask5)
        if self.residual:
            flow_coarse5 = flow_coarse5 + flow5
        #flow_fine5 = self.cnet5(torch.cat([x1_lv5, upfeat5, flow5, mask5], dim=1))
        flow_fine5 = self.cnet5(torch.cat([x1_lv5, flow5, mask5], dim=1))
        flow5 = flow_coarse5 + flow_fine5

        flow4 = F.interpolate(flow5, scale_factor=2, mode='bilinear', align_corners=True) * 2
        mask4 = F.interpolate(mask5, scale_factor=2, mode='bilinear', align_corners=True)

        # warp4 = flow4.unsqueeze(1)
        # warp4 = torch.repeat_interleave(warp4, 9, 1)
        # S1, S2, S3, S4, S5 = warp4.shape
        # warp4 = warp4.view(S1, S2 * S3, S4, S5)
        # warp4 = self.deform4(x2_lv4, warp4)
        # warp4 = (warp4 /16. * F.sigmoid(mask4)) + tradeoff4
        # warp4 = self.relu(warp4)

        warp4 = self.warping_layer(x2_lv4, flow4) * F.sigmoid(mask4) + tradeoff4

        corr4 = self.relu(self.corr(x1_lv4, warp4))
        #tradeoff3, upfeat3, flow_coarse4, mask4 = self.fms4(x1_lv4, upfeat4, corr4, flow4, mask4)
        tradeoff3, _, flow_coarse4, mask4 = self.fms4(x1_lv4, corr4, flow4, mask4)
        if self.residual:
            flow_coarse4 = flow_coarse4 + flow4
        #flow_fine4 = self.cnet4(torch.cat([x1_lv4, upfeat4, flow4, mask4], dim=1))
        flow_fine4 = self.cnet4(torch.cat([x1_lv4, flow4, mask4], dim=1))
        flow4 = flow_coarse4 + flow_fine4

        flow3 = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=True) * 2
        mask3 = F.interpolate(mask4, scale_factor=2, mode='bilinear', align_corners=True)

        # warp3 = flow3.unsqueeze(1)
        # warp3 = torch.repeat_interleave(warp3, 9, 1)
        # S1, S2, S3, S4, S5 = warp3.shape
        # warp3 = warp3.view(S1, S2 * S3, S4, S5)
        # warp3 = self.deform3(x2_lv3, warp3)
        # warp3 = (warp3 /8. * F.sigmoid(mask3)) + tradeoff3
        # warp3 = self.relu(warp3)

        warp3 = self.warping_layer(x2_lv3, flow3) * F.sigmoid(mask3) + tradeoff3

        corr3 = self.relu(self.corr(x1_lv3, warp3))
        #tradeoff2, upfeat2, flow_coarse3, mask3 = self.fms3(x1_lv3, upfeat3, corr3, flow3, mask3)
        tradeoff2, _, flow_coarse3, mask3 = self.fms3(x1_lv3, corr3, flow3, mask3)
        if self.residual:
            flow_coarse3 = flow_coarse3 + flow3
        #flow_fine3 = self.cnet3(torch.cat([x1_lv3, upfeat3, flow3, mask3], dim=1))
        flow_fine3 = self.cnet3(torch.cat([x1_lv3, flow3, mask3], dim=1))
        flow3 = flow_coarse3 + flow_fine3

        flow2 = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=True) * 2
        mask2 = F.interpolate(mask3, scale_factor=2, mode='bilinear', align_corners=True)

        # warp2 = flow2.unsqueeze(1)
        # warp2 = torch.repeat_interleave(warp2, 9, 1)
        # S1, S2, S3, S4, S5 = warp2.shape
        # warp2 = warp2.view(S1, S2 * S3, S4, S5)
        # warp2 = self.deform2(x2_lv2, warp2)
        # warp2 = (warp2 /4. * F.sigmoid(mask2)) + tradeoff2
        # warp2 = self.relu(warp2)

        warp2 = self.warping_layer(x2_lv2, flow2) * F.sigmoid(mask2) + tradeoff2

        corr2 = self.relu(self.corr(x1_lv2, warp2))
        #_, _, flow_coarse2, mask2 = self.fms2(x1_lv2, upfeat2, corr2, flow2, mask2)
        _, _, flow_coarse2, mask2 = self.fms2(x1_lv2, corr2, flow2, mask2)
        if self.residual:
            flow_coarse2 = flow_coarse2 + flow2
        #flow_fine2 = self.cnet2(torch.cat([x1_lv2, upfeat2, flow2, mask2], dim=1))
        flow_fine2 = self.cnet2(torch.cat([x1_lv2, flow2, mask2], dim=1))
        flow2 = flow_coarse2 + flow_fine2

        flow2 = F.interpolate(flow2, scale_factor=2 ** (self.num_levels - self.output_level - 1), mode='bilinear',
                              align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)

        if test_mode == False:
            return [flow6, flow5, flow4, flow3, flow2]
        else:
            mask = F.interpolate(torch.sigmoid(mask2), scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True)
            return mask, flow2

class MaskFlowNet_v2(nn.Module):
    def __init__(self, batch_norm=False):
        super(MaskFlowNet_v2, self).__init__()
        lv_chs = [3, 16, 32, 64, 96, 128, 196]
        search_range = 4
        self.residual = True
        self.num_levels = len(lv_chs)
        self.output_level = 4
        #lv_chs = [3, 16, 32, 64, 96, 128, 196]

        self.relu = nn.LeakyReLU(0.1)
        self.warping_layer = WarpingLayer()

        self.lv1 = FeatureExtractor(3, 16)
        self.lv2 = FeatureExtractor(16, 32)
        self.lv3 = FeatureExtractor(32, 64)
        self.lv4 = FeatureExtractor(64, 96)
        self.lv5 = FeatureExtractor(96, 128)
        self.lv6 = FeatureExtractor(128, 196)

        self.corr = Correlation(pad_size=search_range, kernel_size=1, max_displacement=search_range, stride1=1, stride2=1, corr_multiply=1)

        corr_dim = (search_range * 2 + 1) ** 2  # 9*9

        chs = corr_dim + 196 + 16 + 2 + 1
        #chs = corr_dim + 196 + 2 + 1
        self.fms6 = FlowAndMaskEstimator_v2(chs, 128)

        chs = corr_dim + 128 + 16 + 2 + 1
        #chs = corr_dim + 128 + 2 + 1
        self.fms5 = FlowAndMaskEstimator_v2(chs, 96)

        chs = corr_dim + 96 + 16 + 2 + 1
        #chs = corr_dim + 96 + 2 + 1
        self.fms4 = FlowAndMaskEstimator_v2(chs, 64)

        chs = corr_dim + 64 + 16 + 2 + 1
        #chs = corr_dim + 64 + 2 + 1
        self.fms3 = FlowAndMaskEstimator_v2(chs, 32)

        chs = corr_dim + 32 + 16 + 2 + 1
        #chs = corr_dim + 32 + 2 + 1
        self.fms2 = FlowAndMaskEstimator_v2(chs, 16)

        c_chs = 196 + 16 + 2 + 1
        #c_chs = 196 + 2 + 1
        self.cnet6 = ContextNetwork(c_chs)

        c_chs = 128 + 16 + 2 + 1
        #c_chs = 128 + 2 + 1
        self.cnet5 = ContextNetwork(c_chs)

        c_chs = 96 + 16 + 2 + 1
        #c_chs = 96 + 2 + 1
        self.cnet4 = ContextNetwork(c_chs)

        c_chs = 64 + 16 + 2 + 1
        #c_chs = 64 + 2 + 1
        self.cnet3 = ContextNetwork(c_chs)

        c_chs = 32 + 16 + 2 + 1
        #c_chs = 32 + 2 + 1
        self.cnet2 = ContextNetwork(c_chs)

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

        shape = list(x1_lv6.size());shape[1] = 2
        flow6 = torch.zeros(shape).to(x1_lv6.device)
        shape[1] = 1
        mask6 = torch.ones(shape).to(x1_lv6.device)
        shape[1] = 16
        upfeat6 = torch.zeros(shape).to(x1_lv6.device)

        warp6 = self.warping_layer(x2_lv6, flow6)
        warp6 = self.relu(warp6)

        corr6 = self.relu(self.corr(x1_lv6, warp6))
        _, upfeat5, flow_coarse6, mask6 = self.fms6(x1_lv6, upfeat6, corr6, flow6, mask6)
        #_, _, flow_coarse6, mask6 = self.fms6(x1_lv6, corr6, flow6, mask6)
        if self.residual:
            flow_coarse6 = flow_coarse6 + flow6
        flow_fine6 = self.cnet6(torch.cat([x1_lv6, upfeat6, flow6, mask6], dim=1))  #这里的upfeat加上去还是有用的
        #flow_fine6 = self.cnet6(torch.cat([x1_lv6, flow6, mask6], dim=1))
        flow6 = flow_coarse6 + flow_fine6

        flow5 = F.interpolate(flow6, scale_factor = 2, mode = 'bilinear', align_corners=True) * 2
        mask5 = F.interpolate(mask6, scale_factor = 2, mode = 'bilinear', align_corners=True)

        warp5 = self.warping_layer(x2_lv5, flow5) * F.sigmoid(mask5) #这里能不能加上offset 加到warp后的特征图上再乘mask
        warp5 = self.relu(warp5)

        corr5 = self.relu(self.corr(x1_lv5, warp5))
        _, upfeat4, flow_coarse5, mask5 = self.fms5(x1_lv5, upfeat5, corr5, flow5, mask5)
        #_, _, flow_coarse5, mask5 = self.fms5(x1_lv5, corr5, flow5, mask5)
        if self.residual:
            flow_coarse5 = flow_coarse5 + flow5
        flow_fine5 = self.cnet5(torch.cat([x1_lv5, upfeat5, flow5, mask5], dim=1))
        #flow_fine5 = self.cnet5(torch.cat([x1_lv5, flow5, mask5], dim=1))
        flow5 = flow_coarse5 + flow_fine5

        flow4 = F.interpolate(flow5, scale_factor=2, mode='bilinear', align_corners=True) * 2
        mask4 = F.interpolate(mask5, scale_factor=2, mode='bilinear', align_corners=True)

        warp4 = self.warping_layer(x2_lv4, flow4) * F.sigmoid(mask4)
        warp4 = self.relu(warp4)

        corr4 = self.relu(self.corr(x1_lv4, warp4))
        _, upfeat3, flow_coarse4, mask4 = self.fms4(x1_lv4, upfeat4, corr4, flow4, mask4)
        #_, _, flow_coarse4, mask4 = self.fms4(x1_lv4, corr4, flow4, mask4)
        if self.residual:
            flow_coarse4 = flow_coarse4 + flow4
        flow_fine4 = self.cnet4(torch.cat([x1_lv4, upfeat4, flow4, mask4], dim=1))
        #flow_fine4 = self.cnet4(torch.cat([x1_lv4, flow4, mask4], dim=1))
        flow4 = flow_coarse4 + flow_fine4

        flow3 = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=True) * 2
        mask3 = F.interpolate(mask4, scale_factor=2, mode='bilinear', align_corners=True)

        warp3 = self.warping_layer(x2_lv3, flow3) * F.sigmoid(mask3)
        warp3 = self.relu(warp3)

        corr3 = self.relu(self.corr(x1_lv3, warp3))
        _, upfeat2, flow_coarse3, mask3 = self.fms3(x1_lv3, upfeat3, corr3, flow3, mask3)
        #_, _, flow_coarse3, mask3 = self.fms3(x1_lv3, corr3, flow3, mask3)
        if self.residual:
            flow_coarse3 = flow_coarse3 + flow3
        flow_fine3 = self.cnet3(torch.cat([x1_lv3, upfeat3, flow3, mask3], dim=1))
        #flow_fine3 = self.cnet3(torch.cat([x1_lv3, flow3, mask3], dim=1))
        flow3 = flow_coarse3 + flow_fine3

        flow2 = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=True) * 2
        mask2 = F.interpolate(mask3, scale_factor=2, mode='bilinear', align_corners=True)

        warp2 = self.warping_layer(x2_lv2, flow2) * F.sigmoid(mask2)
        warp2 = self.relu(warp2)

        corr2 = self.relu(self.corr(x1_lv2, warp2))
        _, _, flow_coarse2, mask2 = self.fms2(x1_lv2, upfeat2, corr2, flow2, mask2)
        #_, _, flow_coarse2, mask2 = self.fms2(x1_lv2, corr2, flow2, mask2)
        if self.residual:
            flow_coarse2 = flow_coarse2 + flow2
        flow_fine2 = self.cnet2(torch.cat([x1_lv2, upfeat2, flow2, mask2], dim=1))
        #flow_fine2 = self.cnet2(torch.cat([x1_lv2, flow2, mask2], dim=1))
        flow2 = flow_coarse2 + flow_fine2

        flow2 = F.interpolate(flow2, scale_factor=2 ** (self.num_levels - self.output_level - 1), mode='bilinear',
                              align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)

        if test_mode == False:
            return [flow6, flow5, flow4, flow3, flow2]
        else:
            mask = F.interpolate(torch.sigmoid(mask2), scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True)
            return mask, flow2

class MaskFlowNet_v3(nn.Module):
    def __init__(self, output_level=4, batch_norm=False):
        super(MaskFlowNet_v3, self).__init__()
        lv_chs = [3, 16, 32, 64, 96, 128, 196]
        search_range = 4
        self.residual = True
        self.num_levels = len(lv_chs)
        self.output_level = output_level
        #lv_chs = [3, 16, 32, 64, 96, 128, 196]

        self.relu = nn.LeakyReLU(0.1)
        self.warping_layer = WarpingLayer()

        self.lv1 = FeatureExtractor(3, 16)
        self.lv2 = FeatureExtractor(16, 32)
        self.lv3 = FeatureExtractor(32, 64)
        self.lv4 = FeatureExtractor(64, 96)
        self.lv5 = FeatureExtractor(96, 128)
        self.lv6 = FeatureExtractor(128, 196)

        self.corr = Correlation(pad_size=search_range, kernel_size=1, max_displacement=search_range, stride1=1, stride2=1, corr_multiply=1)

        corr_dim = (search_range * 2 + 1) ** 2  # 9*9

        chs = corr_dim + 196 + 16 + 2 + 1
        #chs = corr_dim + 196 + 2 + 1
        self.fms6 = AttAggFME(196, 16, 128)

        chs = corr_dim + 128 + 16 + 2 + 1
        #chs = corr_dim + 128 + 2 + 1
        self.fms5 = AttAggFME(128, 16, 96)

        chs = corr_dim + 96 + 16 + 2 + 1
        #chs = corr_dim + 96 + 2 + 1
        self.fms4 = AttAggFME(96, 16, 64)

        chs = corr_dim + 64 + 16 + 2 + 1
        #chs = corr_dim + 64 + 2 + 1
        self.fms3 = AttAggFME(64, 16, 32)

        chs = corr_dim + 32 + 16 + 2 + 1
        #chs = corr_dim + 32 + 2 + 1
        self.fms2 = AttAggFME(32, 16, 16)

        #c_chs = 196 + 16 + 2 + 1
        #c_chs = 196 + 2 + 1
        #self.cnet6 = ContextNetwork(c_chs)

        #c_chs = 128 + 16 + 2 + 1
        #c_chs = 128 + 2 + 1
        #self.cnet5 = ContextNetwork(c_chs)

        #c_chs = 96 + 16 + 2 + 1
        #c_chs = 96 + 2 + 1
        #self.cnet4 = ContextNetwork(c_chs)

        #c_chs = 64 + 16 + 2 + 1
        #c_chs = 64 + 2 + 1
        #self.cnet3 = ContextNetwork(c_chs)

        #c_chs = 32 + 16 + 2 + 1
        #c_chs = 32 + 2 + 1
        #self.cnet2 = ContextNetwork(c_chs)

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

        shape = list(x1_lv6.size());shape[1] = 2
        flow6 = torch.zeros(shape).to(x1_lv6.device)
        shape[1] = 1
        mask6 = torch.ones(shape).to(x1_lv6.device)
        shape[1] = 16
        upfeat6 = torch.zeros(shape).to(x1_lv6.device)

        warp6 = self.warping_layer(x2_lv6, flow6)
        warp6 = self.relu(warp6)

        corr6 = self.relu(self.corr(x1_lv6, warp6))
        tradeoff5, upfeat5, flow_coarse6, mask6 = self.fms6(x1_lv6, upfeat6, corr6, flow6)
        #_, _, flow_coarse6, mask6 = self.fms6(x1_lv6, corr6, flow6, mask6)
        if self.residual:
            flow_coarse6 = flow_coarse6 + flow6
        #flow_fine6 = self.cnet6(torch.cat([x1_lv6, upfeat6, flow6, mask6], dim=1))
        #flow_fine6 = self.cnet6(torch.cat([x1_lv6, flow6, mask6], dim=1))
        #flow6 = flow_coarse6 + flow_fine6
        flow6 = flow_coarse6

        flow5 = F.interpolate(flow6, scale_factor = 2, mode = 'bilinear', align_corners=True) * 2
        mask5 = F.interpolate(mask6, scale_factor = 2, mode = 'bilinear', align_corners=True)

        warp5 = self.warping_layer(x2_lv5, flow5)# * F.sigmoid(mask5) + tradeoff5
        warp5 = self.relu(warp5)

        corr5 = self.relu(self.corr(x1_lv5, warp5))
        tradeoff4, upfeat4, flow_coarse5, mask5 = self.fms5(x1_lv5, upfeat5, corr5, flow5)
        #_, _, flow_coarse5, mask5 = self.fms5(x1_lv5, corr5, flow5, mask5)
        if self.residual:
            flow_coarse5 = flow_coarse5 + flow5
        #flow_fine5 = self.cnet5(torch.cat([x1_lv5, upfeat5, flow5, mask5], dim=1))
        #flow_fine5 = self.cnet5(torch.cat([x1_lv5, flow5, mask5], dim=1))
        #flow5 = flow_coarse5 + flow_fine5
        flow5 = flow_coarse5

        flow4 = F.interpolate(flow5, scale_factor=2, mode='bilinear', align_corners=True) * 2
        mask4 = F.interpolate(mask5, scale_factor=2, mode='bilinear', align_corners=True)

        warp4 = self.warping_layer(x2_lv4, flow4)# * F.sigmoid(mask4) + tradeoff4
        warp4 = self.relu(warp4)

        corr4 = self.relu(self.corr(x1_lv4, warp4))
        tradeoff3, upfeat3, flow_coarse4, mask4 = self.fms4(x1_lv4, upfeat4, corr4, flow4)
        #_, _, flow_coarse4, mask4 = self.fms4(x1_lv4, corr4, flow4, mask4)
        if self.residual:
            flow_coarse4 = flow_coarse4 + flow4
        #flow_fine4 = self.cnet4(torch.cat([x1_lv4, upfeat4, flow4, mask4], dim=1))
        #flow_fine4 = self.cnet4(torch.cat([x1_lv4, flow4, mask4], dim=1))
        #flow4 = flow_coarse4 + flow_fine4
        flow4 = flow_coarse4

        flow3 = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=True) * 2
        mask3 = F.interpolate(mask4, scale_factor=2, mode='bilinear', align_corners=True)

        warp3 = self.warping_layer(x2_lv3, flow3)# * F.sigmoid(mask3) + tradeoff3
        warp3 = self.relu(warp3)

        corr3 = self.relu(self.corr(x1_lv3, warp3))
        tradeoff2, upfeat2, flow_coarse3, mask3 = self.fms3(x1_lv3, upfeat3, corr3, flow3)
        #_, _, flow_coarse3, mask3 = self.fms3(x1_lv3, corr3, flow3, mask3)
        if self.residual:
            flow_coarse3 = flow_coarse3 + flow3
        #flow_fine3 = self.cnet3(torch.cat([x1_lv3, upfeat3, flow3, mask3], dim=1))
        #flow_fine3 = self.cnet3(torch.cat([x1_lv3, flow3, mask3], dim=1))
        #flow3 = flow_coarse3 + flow_fine3
        flow3 = flow_coarse3

        if self.output_level == 3:
            flow3 = F.interpolate(flow3, scale_factor=2 ** (self.num_levels - self.output_level - 1), mode='bilinear',
                              align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)
            if test_mode == False:
                return [flow6, flow5, flow4, flow3]
            else:
                mask = F.interpolate(torch.sigmoid(mask3), scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True)
                return mask, flow3

        flow2 = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=True) * 2
        mask2 = F.interpolate(mask3, scale_factor=2, mode='bilinear', align_corners=True)

        warp2 = self.warping_layer(x2_lv2, flow2)# * F.sigmoid(mask2) + tradeoff2
        warp2 = self.relu(warp2)

        corr2 = self.relu(self.corr(x1_lv2, warp2))
        _, _, flow_coarse2, mask2 = self.fms2(x1_lv2, upfeat2, corr2, flow2)
        #_, _, flow_coarse2, mask2 = self.fms2(x1_lv2, corr2, flow2, mask2)
        if self.residual:
            flow_coarse2 = flow_coarse2 + flow2
        #flow_fine2 = self.cnet2(torch.cat([x1_lv2, upfeat2, flow2, mask2], dim=1))
        #flow_fine2 = self.cnet2(torch.cat([x1_lv2, flow2, mask2], dim=1))
        #flow2 = flow_coarse2 + flow_fine2
        flow2 = flow_coarse2

        if self.output_level == 4:
            flow2 = F.interpolate(flow2, scale_factor=2 ** (self.num_levels - self.output_level - 1), mode='bilinear',
                              align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)
            if test_mode == False:
                return [flow6, flow5, flow4, flow3, flow2]
            else:
                mask = F.interpolate(torch.sigmoid(mask2), scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True)
                return mask, flow2
