import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .module import FeatureExtractorRes, PAFB

class Net(nn.Module):
    def __init__(self, fe=32, fm=16, factor=20):
        super(Net, self).__init__()
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.fe1 = FeatureExtractorRes(fe=fe, fm=fm, factor=factor)
        self.fe2 = FeatureExtractorRes(fe=fe, fm=fm, factor=factor)
        self.fe4 = FeatureExtractorRes(fe=fe, fm=fm, factor=factor)
        self.conv1 = nn.Conv2d(3, 3*4, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 3*16, 3, 1, 1)
        self.shuffle2 = nn.PixelShuffle(2)
        self.shuffle4 = nn.PixelShuffle(4)
        self.conv3 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv4 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv5 = nn.Conv2d(3, 3, 3, 1, 1)

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

        l2_u2 = self.shuffle2(self.conv1(l2_align))
        l4_u4 = self.shuffle4(self.conv2(l4_align))

        l1_align = self.relu(self.conv3(l1_align))
        l1_align = self.relu(self.conv4(l1_align))
        l1_align = self.conv5(l1_align)

        l2_u2 = self.relu(self.conv3(l2_u2))
        l2_u2 = self.relu(self.conv4(l2_u2))
        l2_u2 = self.conv5(l2_u2)

        l4_u4 = self.relu(self.conv3(l4_u4))
        l4_u4 = self.relu(self.conv4(l4_u4))
        l4_u4 = self.conv5(l4_u4)

        Ic = torch.cat([l1_align, l2_u2, l4_u4, x2_raw], dim=1)

        out = self.last(Ic)

        return out

if __name__ == '__main__':
    a=torch.rand(1, 3, 320, 320)
    b=torch.rand(1, 3, 320, 320)
    xx = Net()
    out = xx(a, b)
    print(out.shape)

