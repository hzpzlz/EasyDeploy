import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .module import FeatureExtractor, PAFB

class Net(nn.Module):
    def __init__(self, fe=32, fm=16, factor=20):
        super(Net, self).__init__()
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.fe1 = FeatureExtractor(fe=fe, fm=fm, factor=factor)
        self.fe2 = FeatureExtractor(fe=fe, fm=fm, factor=factor)
        self.fe4 = FeatureExtractor(fe=fe, fm=fm, factor=factor)

        self.last = PAFB(fe=16, fm=32)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x1_raw, x2_raw): #x1_raw:ref, x2_raw:target
        x1_d2 = F.interpolate(x1_raw, scale_factor=0.5)
        x2_d2 = F.interpolate(x2_raw, scale_factor=0.5)

        x1_d4 = F.interpolate(x1_d2, scale_factor=0.5)
        x2_d4 = F.interpolate(x2_d2, scale_factor=0.5)

        l1_align = self.fe1(x1_raw, x2_raw)
        l2_align = self.fe2(x1_d2, x2_d2)
        l4_align = self.fe4(x1_d4, x2_d4)

        l2_u2 = F.interpolate(l2_align, scale_factor=2)
        l4_u4 = F.interpolate(l4_align, scale_factor=4)

        Ic = torch.cat([l1_align, l2_u2, l4_u4, x2_raw], dim=1)

        out = self.last(Ic)

        return out

if __name__ == '__main__':
    a=torch.rand(1, 3, 320, 320)
    b=torch.rand(1, 3, 320, 320)
    xx = Net()
    out = xx(a, b)
    print(out.shape)

