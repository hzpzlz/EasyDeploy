import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import FeatureExtractor, FlowEstimator, ContextNetwork, upsample2d
import time

class PWCPlus(nn.Module):
    """docstring for PWCNet"""
    def __init__(self, factor=5):
        super(PWCPlus, self).__init__()
        self.Encoder_lv1 = FeatureExtractor(3, 16)   #240
        self.Encoder_lv2 = FeatureExtractor(16, 32)  #120
        self.Encoder_lv3 = FeatureExtractor(32, 64) #60
        self.Encoder_lv4 = FeatureExtractor(64, 96) #30
        self.Encoder_lv5 = FeatureExtractor(96, 128) #15

        #self.upsample2d = upsample2d

        self.flow_resi5 = FlowEstimator(128, factor)  ##
        self.flow_resi4 = FlowEstimator(96, factor)  ##
        self.flow_resi3 = FlowEstimator(64, factor)  ##

        self.flow_fine = ContextNetwork(64+2)

    def forward(self, x1_raw, x2_raw): #x1_raw:base,x2_raw:ref
        x1_lv3 = self.Encoder_lv3(self.Encoder_lv2(self.Encoder_lv1(x1_raw)))
        x1_lv4 = self.Encoder_lv4(x1_lv3)
        x1_lv5 = self.Encoder_lv5(x1_lv4)

        x2_lv3 = self.Encoder_lv3(self.Encoder_lv2(self.Encoder_lv1(x2_raw)))
        x2_lv4 = self.Encoder_lv4(x2_lv3)
        x2_lv5 = self.Encoder_lv5(x2_lv4)

        tb, tc, th, tw = tuple(int(i) for i in x1_lv5.shape)
        shape = [tb, 2, th, tw]

        flow5 = torch.zeros(shape).to(x1_lv5.device)
        delta_flow5 = self.flow_resi5(x1_lv5, x2_lv5, flow5)
        flow5 += torch.add(flow5, delta_flow5)

        flow4 = F.interpolate(flow5, scale_factor=2, mode='bilinear', align_corners=True)
        delta_flow4 = self.flow_resi4(x1_lv4, x2_lv4, flow4)
        flow4 += torch.add(flow4, delta_flow4)

        flow3 = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=True)
        delta_flow3 = self.flow_resi3(x1_lv3, x2_lv3, flow3)
        flow3 += torch.add(flow3, delta_flow3)

        flow_refine = self.flow_fine(torch.cat((x1_lv3, flow3), dim=1))
        flow = torch.add(flow3, flow_refine)

        flow0 = F.interpolate(flow, scale_factor=8, mode='bilinear', align_corners=True) * 20
        return flow0 

if __name__ == '__main__':
    a = torch.rand(8, 3, 480*3, 480*4).cuda()
    b = torch.rand(8, 3, 480*3, 480*4).cuda()
    print(a.device)
    s=time.time()
    xx = PWCPlus().cuda()
    out = xx(a, b)
    e = time.time()
    print(e - s)
    print(out.shape)

