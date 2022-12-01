import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .modules import WarpingLayer, FeaturePyramidExtractor, CostVolumeLayer, OpticalFlowEstimator, ContextNetwork, DeformableNet
from ..correlation_package.correlation import Correlation


class MaskFlowNet(nn.Module):
    def __init__(self, batch_norm=False):
        super(MaskFlowNet, self).__init__()
        lv_chs = [3, 16, 32, 64, 96, 128, 192]
        search_range = 4
        self.residual = True
        self.num_levels = len(lv_chs)
        self.output_level = 4

        batch_norm = batch_norm

        self.feature_pyramid_extractor = FeaturePyramidExtractor(lv_chs=lv_chs, batch_norm=batch_norm)
        
        self.warping_layer = WarpingLayer()

        self.corr = Correlation(pad_size = search_range, kernel_size = 1, max_displacement = search_range, stride1 = 1, stride2 = 1, corr_multiply = 1)
        
        #self.flow_estimators = []
        self.flow_estimators = nn.ModuleList()
        for l, (ch, ch_f) in enumerate(zip(lv_chs[::-1], lv_chs[:-1][::-1])):
            layer = OpticalFlowEstimator(ch + (search_range*2+1)**2 + 2, ch_f, batch_norm=batch_norm)
            #self.add_module(f'FlowEstimator(Lv{l})', layer)
            self.flow_estimators.append(layer)

        self.deforms = nn.ModuleList()
        for l, ch in enumerate(lv_chs[::-1]):
            layer = DeformableNet(ch, batch_norm=batch_norm)
            #self.add_module(f'DeformConv(Lv{l})', layer)
            self.deforms.append(layer)

        self.context_networks = nn.ModuleList()
        for l, ch in enumerate(lv_chs[::-1]):
            layer = ContextNetwork(ch + 2, batch_norm=batch_norm)
            #self.add_module(f'ContextNetwork(Lv{l})', layer)
            self.context_networks.append(layer)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x1_raw, x2_raw, test_mode=False):
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        flows = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # upsample flow and scale the displacement
            if l == 0:
                shape = list(x1.size()); shape[1] = 2
                flow = torch.zeros(shape).to(x1.device)
                mask = torch.ones(shape).to(x1.device)
                x2_warp = x2 #self.warping_layer(x2, flow)
            else:
                flow = F.interpolate(flow, scale_factor = 2, mode = 'bilinear', align_corners=True) * 2
                mask = F.interpolate(mask, scale_factor = 2, mode = 'bilinear', align_corners=True)

                warp = flow.unsqueeze(1)
                warp = torch.repeat_interleave(warp, 9, 1)
                S1, S2, S3, S4, S5 = warp.shape
                warp = warp.view(S1, S2 * S3, S4, S5)
                warp = self.deforms[l](x2, warp)
                warp = (warp * torch.sigmoid(mask)) + upfeat
                x2_warp = F.leaky_relu_(warp)
            
            # correlation
            corr = self.corr(x1, x2_warp)
            corr = F.leaky_relu_(corr)

            # concat and estimate flow
            # ATTENTION: `+ flow` makes flow estimator learn to estimate residual flow
            if self.residual:
                upfeat, flow_coarse, mask = self.flow_estimators[l](torch.cat([x1, corr, flow], dim = 1))
                flow_coarse = flow_coarse + flow
            else:
                upfeat, flow_coarse, mask = self.flow_estimators[l](torch.cat([x1, corr, flow], dim = 1))
            
            flow_fine = self.context_networks[l](torch.cat([x1, flow], dim = 1))
            flow = flow_coarse + flow_fine

            if l == self.output_level:
                flow = F.interpolate(flow, scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)
                flows.append(flow)
                break
            else:
                flows.append(flow)
        if test_mode == False:
            return flows
        else:
            mask = F.interpolate(torch.sigmoid(mask), scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True)
            return mask, flows[-1]
