import torch
import torch.nn as nn
from .modules import FeatureExtractor, FlowEstimator, ContextNetwork, upsample2d
import time

class PWCNet(nn.Module):
    """docstring for PWCNet"""
    def __init__(self):
        super(PWCNet, self).__init__()

        #### FeatureExtractor
        self.Encoder_lv1 = FeatureExtractor(6, 16)
        self.Encoder_lv2 = FeatureExtractor(16, 32)
        self.Encoder_lv3 = FeatureExtractor(32, 64)
        self.Encoder_lv4 = FeatureExtractor(64, 96)
        self.Encoder_lv5 = FeatureExtractor(96, 128)
        self.Encoder_lv6 = FeatureExtractor(128, 196)

        #### upsample function
        self.upsample2d = upsample2d

        ####  FlowEstimator (no densenet)
        ####  dim = dim(feature1) + dim(corr) + 2(up_flow)
        self.flow_resi6 = FlowEstimator(196)  ## 196 + 81
        self.flow_resi5 = FlowEstimator(130)  ## 128 + 81 + 2
        self.flow_resi4 = FlowEstimator( 98)  ## 96 +  81 + 2
        self.flow_resi3 = FlowEstimator( 66)  ## 64 +  81 + 2

        ####  ContextNetwork (no densenet and every level with contextnet)
        ####  dim = dim(feature1) + 2(up_flow)
        self.flow_fine = ContextNetwork(66)  ## 32 + 2

        #### init weight and bias in network
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #        nn.init.kaiming_normal_(m.weight, 0.1)
        #        if m.bias is not None:
        #            nn.init.constant_(m.bias, 0)
        #    elif isinstance(m, nn.LeakyReLU) or isinstance(m, nn.Sequential):
        #        pass

    def forward(self, x1_raw, x2_raw, test_mode=False):

        x_raw = torch.cat((x1_raw, x2_raw), dim=1)

        #### extract feature by encoder
        x_lv3 = self.Encoder_lv3(self.Encoder_lv2(self.Encoder_lv1(x_raw)))
        x_lv4 = self.Encoder_lv4(x_lv3)
        x_lv5 = self.Encoder_lv5(x_lv4)
        x_lv6 = self.Encoder_lv6(x_lv5)


        #### optical flow estimation
        #### flow6
        flow6 = self.flow_resi6(x_lv6)

        #### flow5
        flow5_up = self.upsample2d(flow6, x_lv5)
        flow5 = self.flow_resi5(torch.cat((x_lv5, flow5_up), dim=1))
        flow5 = torch.add(flow5, flow5_up)

        #### flow4
        flow4_up = self.upsample2d(flow5, x_lv4)
        flow4 = self.flow_resi4(torch.cat((x_lv4, flow4_up), dim=1))
        flow4 = torch.add(flow4, flow4_up)

        #### flow3
        flow3_up = self.upsample2d(flow4, x_lv3)
        flow_corse3 = self.flow_resi3(torch.cat((x_lv3, flow3_up), dim=1))
        flow_corse3 = torch.add(flow_corse3, flow3_up)

        #### flow refine
        flow_fine3 = self.flow_fine(torch.cat((x_lv3, flow_corse3), dim=1))
        flow3 = torch.add(flow_corse3, flow_fine3)

        #### flow0
        flow0 = self.upsample2d(flow3, x1_raw) * 20.0
        #flow0 = self.upsample2d(flow3, x1_raw)

        return flow0

if __name__ == '__main__':
    a = torch.rand(8, 3, 480*3, 480*4).cuda()
    b = torch.rand(8, 3, 480*3, 480*4).cuda()
    print(a.device)
    s=time.time()
    xx = PWCNet().cuda()
    out = xx(a, b)
    e = time.time()
    print(e - s)
    print(out.shape)
