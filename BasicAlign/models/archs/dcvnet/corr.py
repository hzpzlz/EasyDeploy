import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class corrs8(nn.Module):
    def __init__(self, in_chans, out_chans, rate=1, k=4):
        super(corrs8, self).__init__()
        self.k = k
        radius = [1, 3, 5, 9, 13, 21]
        self.dila_convs = nn.ModuleList()
        for i in radius:
            self.dila_convs.append(nn.Conv2d(in_chans, out_chans, 3, 1, padding=i*rate, dilation=i*rate, bias=True))

    def forward(self, x, y):
        b,c,h,w = y.size()
        k = self.k
        y = F.pad(y, (k,k,k,k), 'reflect')
        ys = [conv(y) for conv in self.dila_convs]

        result = []
        for i in np.arange(-k, k+1, 1):
            for j in np.arange(-k, k+1, 1):
                sx = k+i
                sy = k+j
                corr = [torch.cosine_similarity(x.view(b, 4, -1, h, w), yy[:, :, sx:sx+h, sy:sy+w].view(b, 4, -1, h, w), dim=2) for yy in ys]
                corrs = torch.cat(corr, dim=1)
                result.append(corrs)

        out = torch.stack(result, dim=2)
        #print(out.shape)
        return out

class corrs2(nn.Module):
    def __init__(self, in_chans, out_chans, k=4):
        super(corrs2, self).__init__()
        self.k = k
        self.dila_conv = nn.Conv2d(in_chans, out_chans, 3, 1, padding=1, dilation=1, bias=True)

    def forward(self, x, y):
        b,c,h,w = y.size()
        x = F.interpolate(x, (h//4, w//4), None, 'bilinear', True)
        y = F.interpolate(y, (h // 4, w // 4), None, 'bilinear', True)
        _, _, h, w = y.size()

        k = self.k
        y = F.pad(y, (k,k,k,k), 'reflect')
        y = self.dila_conv(y)

        result = []
        for i in np.arange(-k, k+1, 1):
            for j in np.arange(-k, k+1, 1):
                sx = k+i
                sy = k+j
                corr = torch.cosine_similarity(x.view(b, 4, -1, h, w), y[:, :, sx:sx+h, sy:sy+w].view(b, 4, -1, h, w), dim=2)
                result.append(corr)

        out = torch.stack(result, dim=2)
        #print(out.shape)
        return out


if __name__ == '__main__':
    pass
    #print(out.permute(0, 2, 1, 3, 4).shape)




