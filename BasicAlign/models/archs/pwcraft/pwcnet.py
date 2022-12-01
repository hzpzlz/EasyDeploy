import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .modules import WarpingLayer, FeaturePyramidExtractor, CostVolumeLayer, OpticalFlowEstimator, ContextNetwork, FlowEstimatorDense
from ..correlation_package.correlation import Correlation
from .raft import CorrBlock, SmallUpdateBlock

class Netv3(nn.Module):
    def __init__(self, lv_chs, search_range, residual, output_level, batch_norm=False):
        super(Netv3, self).__init__()
        lv_chs = lv_chs
        self.search_range = search_range
        self.residual = residual
        self.num_levels = len(lv_chs)
        self.output_level = output_level
        self.iters = [12, 10, 8, 6, 4, 2, 1]

        self.feature_pyramid_extractor = FeaturePyramidExtractor(lv_chs=lv_chs, batch_norm=batch_norm)
        
        self.warping_layer = WarpingLayer()

        self.corr_list = []
        #self.corr_dim = []
        for l, ch in enumerate(lv_chs[::-1]):
            corr = CorrBlock(num_levels=l, radius=search_range)
            #self.corr_dim.append((l+1) * (2*search_range+1)**2)
            self.corr_list.append(corr)

        self.update_list = []
        for l, ch in enumerate(lv_chs[::-1]):
            update = SmallUpdateBlock(corr_levels=l+1, corr_radius=search_range, hidden_dim=ch)
            self.add_module(f'UpdateFlow(Lv{l})', update)
            self.update_list.append(update)
        
        self.flow_estimators = []
        for l, ch in enumerate(lv_chs[::-1]):
            layer = OpticalFlowEstimator(ch + (search_range*2+1)**2 + 2, batch_norm=batch_norm)
            self.add_module(f'FlowEstimator(Lv{l})', layer)
            self.flow_estimators.append(layer)
        
        self.context_networks = []
        for l, ch in enumerate(lv_chs[::-1]):
            layer = ContextNetwork(ch + 2, batch_norm=batch_norm)
            self.add_module(f'ContextNetwork(Lv{l})', layer)
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
                coord1 = coord0 = torch.zeros(shape).cuda()
                net = x1
                for _ in range(self.iters[l]):
                    corr = self.corr_list[l](x1, x2, coord1)
                    flow_iter = coord1 - coord0
                    net, delta_flow = self.update_list[l](net, corr, flow_iter)
                    coord1 = coord1 + delta_flow
                if self.residual:
                    flow_coarse = coord1
                else:
                    flow_coarse = coord1 - coord0
            else:
                flow = F.interpolate(flow, scale_factor = 2, mode = 'bilinear', align_corners=True) * 2
                coord1 = coord0 = flow
                net = x1
                for _ in range(self.iters[l]):
                    corr = self.corr_list[l](x1, x2, coord1)
                    flow_iter = coord1 - coord0
                    net, delta_flow = self.update_list[l](net, corr, flow_iter)
                    coord1 = coord1 + delta_flow
                if self.residual:
                    flow_coarse = coord1
                else:
                    flow_coarse = coord1 - coord0

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
            return flows[-1], flows[-1]

