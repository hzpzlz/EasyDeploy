import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock, AlternateCorrBlock
from .utils import bilinear_sampler, coords_grid, upflow8, get_4p, DLT_solve, get_grid
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class HRAFT(nn.Module):
    def __init__(self, opt):
        super(HRAFT, self).__init__()
        self.small = opt['network_G']['small']
        self.iters = opt['network_G']['iters']
        self.dropout= opt['network_G']['dropout']
        self.alternate_corr = opt['network_G']['alternate_corr']
        self.mixed_precision = opt['network_G']['mixed_precision']
        

        if self.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.corr_levels = 4
            self.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.corr_levels = 4
            self.corr_radius = 4

        # feature network, context network, and update block
        if self.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=self.dropout)
            self.update_block = SmallUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=self.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        ### for theta
        self.block = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),

        )
        self.avg_pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(4*4*128, 8)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, global_flag=False):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        if global_flag==True:
            coords0 = coords_grid(N, H, W).to(img.device)
            coords1 = coords_grid(N, H, W).to(img.device)
        else:
            coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
            coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def downflow8(self, flow, mode='bilinear'):
        new_size = (flow.shape[2] // 8, flow.shape[3] // 8)
        return F.interpolate(flow, size=new_size, mode=mode, align_corners=True) / 8.0

    def forward(self, image1, image2, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])    #feature encoder
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        x = self.block(torch.cat([fmap1, fmap2], 1))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        theta = self.fc(x)

        h4p = get_4p(image2)
        #print(h4p.shape, "444444444444")
        H_mat = DLT_solve(h4p, theta).squeeze(1)
        out_size = image2.shape[-2:]
        grid_global, pred_I2 = get_grid(image2, H_mat, out_size)  #?????????grid

        pred_I2_fea = self.fnet(pred_I2)

        flow_global = grid_global - self.initialize_flow(image1 ,True)[0]

        if self.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        with autocast(enabled=self.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)  #context?????????????????????????????????GRU
            inp = torch.relu(inp)  #inp??? 0?????????????????? context?????? ?????????

        coords0, coords1 = self.initialize_flow(image1)  #???????????????

        #if flow_init is not None:
        coords1 = self.downflow8(flow_global) + coords1

        flow_predictions = []
        for itr in range(self.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume  ???corr???????????? ???????????????L????????????

            flow = coords1 - coords0  #Flow is represented as difference between two coordinate grids flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow  #????????? 0->????????? ??????

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions, [fmap1, pred_I2_fea, fmap2]
