import torch
import torch.nn as nn
import torch.nn.functional as F
from module import FeatureExtractorRes, res_block, conv_pixshuffle

class Net(nn.Module):
    def __init__(self, fe=32, fm=16, factor=20, n_refs=5, ch_in=4):
        super(Net, self).__init__()
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        #self.down = nn.Upsample(scale_factor=0.5)

        self.rb1 = res_block(in_ch=ch_in * n_refs, mid_ch=16)
        self.d1 = nn.Conv2d(ch_in * n_refs + ch_in, ch_in * n_refs, 3, 2, 1, bias=False)
        #self.d1_base = nn.Conv2d(ch_in, ch_in, 3, 2, 1, bias=False)

        self.fe2 = FeatureExtractorRes(fe=fe, fm=fm, factor=factor, ch_in=ch_in * n_refs)
        self.rb2 = res_block(in_ch=ch_in * n_refs, mid_ch=32)
        self.d2 = nn.Conv2d(ch_in * n_refs + ch_in, ch_in * n_refs, 3, 2, 1, bias=False)
        #self.d2_base = nn.Conv2d(ch_in, ch_in, 3, 2, 1, bias=False)

        self.fe3 = FeatureExtractorRes(fe=fe, fm=fm, factor=factor, ch_in=ch_in * n_refs)
        self.rb3 = res_block(in_ch=ch_in * n_refs, mid_ch=64)
        self.d3 = nn.Conv2d(ch_in * n_refs + ch_in, ch_in * n_refs, 3, 2, 1, bias=False)
        #self.d3_base = nn.Conv2d(ch_in, ch_in, 3, 2, 1, bias=False)

        self.fe4 = FeatureExtractorRes(fe=fe, fm=fm, factor=factor, ch_in=ch_in * n_refs)
        self.rb4 = res_block(in_ch=ch_in * n_refs, mid_ch=128)

        self.up4 = conv_pixshuffle(in_ch=ch_in * n_refs, out_ch=ch_in * n_refs)

        self.uprb3 = res_block(in_ch=ch_in * n_refs * 2, mid_ch=64)
        self.up3 = conv_pixshuffle(in_ch=ch_in * n_refs * 2, out_ch=ch_in * n_refs)

        self.uprb2 = res_block(in_ch=ch_in * n_refs * 2, mid_ch=32)
        self.up2 = conv_pixshuffle(in_ch=ch_in * n_refs * 2, out_ch=ch_in * n_refs)

        self.uprb1 = res_block(in_ch=ch_in * n_refs * 2, mid_ch=16)

        self.out = nn.Conv2d(ch_in * n_refs * 2, ch_in, 3, 1, 1, bias=False)

        ## init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x1_raw, x2_raw):
        ##x1_raw:ref_stacks, x2_raw:target
        #3x1_raw: n h w 4*5  x2_raw: n h w 4
        x11 = self.rb1(x1_raw)
        x12 = x2_raw

        x21 = self.d1(torch.cat([x11, x12], 1))
        x22 = F.interpolate(x12, scale_factor=0.5)
        #x22 = self.down(x12)
        x2_align = self.fe2(x21, x22)
        x21 = self.rb2(x2_align)

        x31 = self.d2(torch.cat([x21, x22], 1))
        x32 = F.interpolate(x22, scale_factor=0.5)
        #x32 = self.down(x22)
        x3_align = self.fe3(x31, x32)
        x31 = self.rb3(x3_align)

        x41 = self.d3(torch.cat([x31, x32], 1))
        x42 = F.interpolate(x32, scale_factor=0.5)
        #x42 = self.down(x32)
        x4_align = self.fe4(x41, x42)
        x41 = self.rb4(x4_align)

        x4_up = self.up4(x41)
        x3_cat = torch.cat([x31, x4_up], 1)
        x3_cat = self.uprb3(x3_cat)

        x3_up = self.up3(x3_cat)
        x2_cat = torch.cat([x21, x3_up], 1)
        x2_cat = self.uprb2(x2_cat)

        x2_up = self.up2(x2_cat)
        x1_cat = torch.cat([x11, x2_up], 1)
        x1_cat = self.uprb1(x1_cat)

        out = self.out(x1_cat)

        return out

if __name__ == '__main__':
    a=torch.rand(1, 4*5, 320, 320)
    b=torch.rand(1, 4, 320, 320)
    xx = Net()
    out = xx(a, b)
    print(out.shape)

