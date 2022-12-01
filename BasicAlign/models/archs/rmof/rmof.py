import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock_stn, AlternateCorrBlock
from .utils import bilinear_sampler, coords_grid, upflow8, get_4_pts, create_grid, warp

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


class RMOF(nn.Module):
    def __init__(self, opt):
        super(RMOF, self).__init__()
        self.small = opt['network_G']['small']
        self.iters = opt['network_G']['iters']
        self.dropout= opt['network_G']['dropout']
        self.alternate_corr = opt['network_G']['alternate_corr']
        self.mixed_precision = opt['network_G']['mixed_precision']
        self.mesh_num = opt['network_G']['mesh_num']
        

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

        #if 'dropout' not in self.args:
        #    self.args.dropout = 0

        #if 'alternate_corr' not in self.args:
        #    self.args.alternate_corr = False

        # feature network, context network, and update block
        if self.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=self.dropout)
            self.update_block = SmallUpdateBlock(hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=self.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        self.conv_out = nn.Conv2d(4, 2, 3, padding=1)
        #self.conv_out = nn.Conv2d(2, 2, 3, padding=1)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def down_and_convert(self, x, y, mode='bilinear'):
        new_size = (x.shape[1] // 8, x.shape[2] // 8)
        x = (x + 1.0) / 2 * x.shape[2]
        y = (y + 1.0) / 2 * x.shape[1]
        flow = torch.cat([x, y], -1).permute(0, 3, 1, 2)

        down = F.interpolate(flow, size=new_size, mode=mode, align_corners=True) / 8
        return down


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

        # run the context network
        with autocast(enabled=self.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)  #context的输出被拆成两部分输入GRU
            inp = torch.relu(inp)  #inp是 0下面那条虚线 context特征 不更新

        coords0, coords1 = self.initialize_flow(image1)  #初始化光流

        #if flow_init is not None:
        #    coords1 = coords1 + flow_init

        flow_predictions = []
        img_warp = []
        for itr in range(self.iters):

            delta_flow = coords1 - coords0  #Flow is represented as difference between two coordinate grids flow = coords1 - coords0
            fus_fea = torch.cat([fmap1, fmap2], 1)
            #print(net)
            with autocast(enabled=self.mixed_precision):
                net, mesh_flow, delta_flow = self.update_block(net, inp, fus_fea, delta_flow)
            _, theta = get_4_pts(mesh_flow, self.mesh_num, image1.shape[0])
            x, y = create_grid(theta, image2, self.mesh_num)
            grid = torch.cat([x, y], -1)

            img2 = F.grid_sample(image2, grid, align_corners=True) #后面可以用warp函数代替
            img_warp.append(img2)
            H_flow = self.down_and_convert(x, y)
            #print(H_flow.permute(0, 2, 3, 1))
            coords1 = delta_flow + H_flow

            #print(flow.permute(0, 2, 3, 1))
            with torch.no_grad():
                fmap2 = self.fnet(img2)

            final_flow = upflow8(coords1 - coords0)
            flow_predictions.append(final_flow)

        if test_mode:
            return coords1-coords0, final_flow
            
        return flow_predictions, img_warp
