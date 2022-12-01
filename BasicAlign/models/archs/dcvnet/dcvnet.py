import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

#from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import corrs8, corrs2
from .transform import transformer, upflow8
#from .corr import CorrBlock, AlternateCorrBlock
#from .utils import bilinear_sampler, coords_grid, upflow8

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


class DCVNet(nn.Module):
    def __init__(self):
        super(DCVNet, self).__init__()
        #self.small = opt['network_G']['small']
        #self.iters = opt['network_G']['iters']
        self.dropout= 0#opt['network_G']['dropout']
        #self.alternate_corr = opt['network_G']['alternate_corr']
        self.k = 4
        self.D_dim=7  #len(dila)+1
        self.C_dim=4 #
        self.mixed_precision = False#opt['network_G']['mixed_precision']

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
        #self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout)
        #self.dconv = nn.Conv2d(256, 256, 1, 1, padding=0, dilation=1, bias=True)
        self.corrs8 = corrs8(256, 256)
        self.corrs2 = corrs2(128, 128)
        self.transform = transformer(256, self.k, self.D_dim*self.C_dim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, image1, image2, test_mode=None):
        """ Estimate optical flow between pair of frames """
        #image1 = 2 * (image1 / 255.0) - 1.0
        #image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature network
        with autocast(enabled=self.mixed_precision):
            fmaps2, fmaps8 = self.fnet([image1, image2])    #feature encoder

        fs2map1 = fmaps2[0].float()
        fs2map2 = fmaps2[1].float()
        fs8map1 = fmaps8[0].float()
        fs8map2 = fmaps8[1].float()

        # calculate corr
        corrs2 = self.corrs2(fs2map1, fs2map2)
        corrs8 = self.corrs8(fs8map1, fs8map2)

        corrs_all = torch.cat([corrs8, corrs2], dim=1)  #b 28(7*4) 81(9*9(2*4=1)) h//8 w//8
        b, c, t, h, w = corrs_all.shape

        weight = self.transform(torch.cat([fs8map1, fs8map2], dim=1))
        flow = torch.matmul(weight, corrs_all.permute(0, 3, 4, 1, 2).reshape(b, h, w, -1, 1))

        flow_up = upflow8(flow)

        return flow, flow_up

if __name__ == '__main__':
    x = torch.rand(1, 3, 1200, 1200).cuda()
    y = torch.rand(1, 3, 1200, 1200).cuda()
    net = DCVNet().cuda().eval()
    t0 = time.time()
    with torch.no_grad():
        a = net(x, y)
    t1 = time.time()
    print(t1- t0)
    #print(a.shape, b.shape, c.shape, d.shape)
