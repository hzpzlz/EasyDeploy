import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class res_block(nn.Module):
    def __init__(self, in_ch=20, mid_ch=20, bias=False):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(mid_ch, in_ch, 3, 1, 1, bias=bias)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))

        return x + x2


class conv_pixshuffle(nn.Module):
    def __init__(self, in_ch=20, out_ch=20, scale=2, bias=False):
        super(conv_pixshuffle, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * scale**2, 3, 1, 1, bias=bias)
        self.pixshuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        out = self.pixshuffle(x)

        return out

#hw空间信息转为batch维度
def t2b(x, factor):
    n,c,h,w = x.shape
    assert(h%factor==0 and w%factor==0)
    x = x.reshape([n, c, factor, h//factor, factor, w//factor])
    x = x.permute([0, 2, 4, 1, 3, 5])
    x = x.reshape([n*factor*factor, c, h//factor, w//factor])
    return x

#batch维度转为hw空间信息
def b2t(x, factor):
    n,c,h,w = x.shape
    assert(n%(factor*factor)==0)
    x = x.reshape([n//(factor**2), factor, factor, c, h, w])
    x = x.permute([0, 3, 1, 4, 2, 5])
    x = x.reshape([n//(factor**2), c, h*factor, w*factor])

    return x

class FeatureExtractorRes(nn.Module):
    def __init__(self, fe=16, fm=32, factor=16, ch_in=4*6):
        super(FeatureExtractorRes, self).__init__()
        self.conv1_1 = nn.Conv2d(ch_in, fe, 3, 1, 1, bias=True)
        self.conv1_2 = nn.Conv2d(4, fe, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(fe, fe, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(fe, fm, 1, 1, 0, bias=True)
        self.relu = nn.LeakyReLU(0.1,inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.factor = factor

    def forward(self, x1, x2):  #x1:ref x2:base
        x11 = self.relu(self.conv1_1(x1))
        x12 = self.relu(self.conv2(x11))
        x13 = x12 + x11
        x14 = self.conv3(x13)
        x14 = t2b(x14, self.factor)

        x21 = self.relu(self.conv1_2(x2))
        x22 = self.relu(self.conv2(x21))
        x23 = x22 + x21
        x24 = self.conv3(x23)
        x24 = t2b(x24, self.factor)

        b, c, h, w = x14.shape
        x14 = x14.reshape([b, c, h*w])
        x24 = x24.reshape([b, c, h*w])

        coor = torch.matmul(x24.permute([0, 2, 1]), x14)
        coor = self.softmax(coor)

        v = t2b(x1, self.factor)
        bb, cc, hh, ww = v.shape
        v = v.reshape([bb, cc, hh*ww])

        res = torch.matmul(v, coor)
        res = res.reshape(bb, cc, hh, ww)

        out = b2t(res, self.factor)

        return out + x1

class PAFBRAW_n5(nn.Module):
    def __init__(self, fe=16, fm=32, ch_in=20):
        super(PAFBRAW_n5, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, fe, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(fe, fm, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(fm, 4, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(0.1,inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))

        mask = self.softmax(x3)
        out = mask[:, 0:1, :, :] * x[:, 0:4, :, :] + mask[:, 1:2, :, :] * x[:, 4:8, :, :] + mask[:, 2:3, :, :] * x[:, 8:12, :, :] + mask[:, 3:4, :, :] * x[:, 12:16, :, :]
        return out

if __name__ == '__main__':
    #a = torch.rand(16, 20, 320, 320)
    #b = torch.rand(16, 4, 320, 320)
    #xx = FeatureExtractorRes(ch_in=20)
    #out = xx(a, b)
    #print(out.shape)
    a = torch.rand(16, 20, 320, 320)
    xx = res_block()
    b =xx(a)
    print(b.shape)





