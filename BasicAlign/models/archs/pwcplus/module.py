import torch
import torch.nn as nn
import torch.nn.functional as F

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

def space2depth(x, factor):
    n, c, h, w = x.shape
    assert (h % factor == 0 and w % factor == 0)
    x = x.reshape([n, c, h//factor, factor, w//factor, factor])
    x = x.permute([0, 1, 3, 5, 2, 4])
    x = x.reshape([n, c*factor**2, h//factor, w//factor])
    return x

def depth2space(x, factor):
    n, c, h, w = x.shape
    assert(c%(factor*factor)==0)
    x = x.reshape([n, c//(factor**2), factor, factor, h, w])
    x = x.permute([0, 1, 4, 2, 5, 3])
    x = x.reshape([n, c//(factor**2), h*factor, w*factor])
    return x

def conv(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, dilation = 1):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size = kernel_size, stride = stride,
            padding = padding, dilation = dilation, bias = True),
        nn.LeakyReLU(0.1, inplace=True))

def predict_flow(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1, bias = True))

def upsample2d(src, tar, mode = 'bilinear'):
    return F.interpolate(src, size=tuple(int(i) for i in tar.shape[2:]), mode=mode, align_corners=True)

class FeatureExtractor(nn.Module):
    """docstring for FeatureExtractor"""
    def __init__(self, ch_in, ch_out):
        super(FeatureExtractor, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in,  ch_out, 3, 2),
            conv(ch_out, ch_out, 3, 1)
        )

    def forward(self, x):
        return self.convs(x)

class ContextNetwork(nn.Module):
    """docstring for ContextNetwork"""
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, padding =  1, dilation =  1),
            conv(128,   128, 3, 1, padding =  2, dilation =  2),
            conv(128,   128, 3, 1, padding =  4, dilation =  4),
            conv(128,    96, 3, 1, padding =  8, dilation =  8),
            conv(96,     64, 3, 1, padding = 16, dilation = 16),
            conv(64,     32, 3, 1, padding =  1, dilation =  1),
            predict_flow(32)
        )

    def forward(self, x):
        return self.convs(x)

class FlowEstimator(nn.Module):
    """docstring for FlowEstimator"""
    def __init__(self, ch_in=16, factor=16, fe=16, fm=32):
        super(FlowEstimator, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, fe, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(fe, fe, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(fe, fm, 1, 1, 0, bias=False)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.factor = factor
        self.out_conv1 = nn.Conv2d(ch_in*2+2, 128, 3, 1, 1, bias=True)
        self.out_conv2 = nn.Conv2d(128, 96, 3, 1, 1, bias=True)
        self.out_conv3 = nn.Conv2d(96, 64, 3, 1, 1, bias=True)
        self.out_conv4 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.predict_flow = predict_flow(32)

    def forward(self, x1, x2, flow):  #x1:base x2:ref
        x11 = self.relu(self.conv1(x1))
        x12 = self.relu(self.conv2(x11))
        x13 = x12 + x11
        x14 = self.conv3(x13)
        x14 = t2b(x14, self.factor)

        x21 = self.relu(self.conv1(x2))
        x22 = self.relu(self.conv2(x21))
        x23 = x22 + x21
        x24 = self.conv3(x23)
        x24 = t2b(x24, self.factor)

        b, c, h, w = x14.shape
        x14 = x14.reshape([b, c, h * w])
        x24 = x24.reshape([b, c, h * w])

        corr = torch.matmul(x24.permute([0, 2, 1]), x14)
        corr = self.softmax(corr)

        v = t2b(x2, self.factor)
        bb, cc, hh, ww = v.shape
        v = v.reshape([bb, cc, hh * ww])

        res = torch.matmul(v, corr)
        res = res.reshape(bb, cc, hh, ww)
        
        out = b2t(res, self.factor)+x2
        out = torch.cat([x1, out, flow], 1)
        out = self.relu(self.out_conv1(out))
        out = self.relu(self.out_conv2(out))
        out = self.relu(self.out_conv3(out))
        out = self.relu(self.out_conv4(out))

        flow = self.predict_flow(out)

        return flow

if __name__ == '__main__':
    a = torch.rand(8, 16, 96, 96)
    b = torch.rand(8, 16, 96, 96)
    flow = torch.rand(8, 2, 96, 96)
    xx = FlowEstimator()
    out = xx(a, b, flow)
    print(out.shape)
