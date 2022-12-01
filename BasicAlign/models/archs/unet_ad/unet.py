import torch
import torch.nn as nn
from .module import FeatureExtractorRes, res_block, conv_pixshuffle, PAFBRAW_n5

class Net(nn.Module):
    def __init__(self, fe=32, fm=16, factor=20, n_refs=5, ch_in=4):
        super(Net, self).__init__()
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.rb1 = res_block(in_ch=ch_in*(n_refs + 1), mid_ch=16)
        self.d1 = nn.Conv2d(ch_in * (n_refs + 1), 32, 3, 2, 1, bias=False)

        self.rb2 = res_block(in_ch=32, mid_ch=32)
        self.d2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)

        self.rb3 = res_block(in_ch=64, mid_ch=64)
        self.d3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.rb4 = res_block(in_ch=128, mid_ch=128)

        self.up4 = conv_pixshuffle(128, 64, scale=2)
        self.uprb3 = res_block(in_ch=64, mid_ch=64)
        self.conv3 = nn.Conv2d(128, 64, 3, 1, 1, bias=False)

        self.up3 = conv_pixshuffle(64, 32, scale=2)
        self.uprb2 = res_block(in_ch=32, mid_ch=32)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)

        self.up2 = conv_pixshuffle(32, 16, scale=2)
        self.uprb1 = res_block(in_ch=16, mid_ch=16)
        self.conv1 = nn.Conv2d(ch_in*(n_refs + 1) + 16, 16, 3, 1, 1, bias=False)

        self.conv_out = nn.Conv2d(16, ch_in, 3, 1, 1, bias=False)


    def forward(self, x1_raw, x2_raw):
        x = torch.cat([x1_raw, x2_raw], 1)
        x1 = self.rb1(x)

        x2 = self.d1(x1)
        x2 = self.rb2(x2)

        x3 = self.d2(x2)
        x3 = self.rb3(x3)

        x4 = self.d3(x3)
        x4 = self.rb4(x4)

        up3 = self.up4(x4)
        x3_cat = torch.cat([x3, up3], 1)
        x3_cat = self.relu(self.conv3(x3_cat))

        up2 = self.up3(x3_cat)
        x2_cat = torch.cat([x2, up2], 1)
        x2_cat = self.relu(self.conv2(x2_cat))

        up1 = self.up2(x2_cat)
        x1_cat = torch.cat([x1, up1], 1)
        x1_cat = self.relu(self.conv1(x1_cat))

        out = self.conv_out(x1_cat)

        return out

if __name__ == '__main__':
    a=torch.rand(1, 4*5, 320, 320)
    b=torch.rand(1, 4, 320, 320)
    xx = Net()
    out = xx(a, b)
    print(out.shape)

