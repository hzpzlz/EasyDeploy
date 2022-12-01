import torch
import torch.nn as nn
import torch.nn.functional as F

class transformer(nn.Module):

    def __init__(self, in_chans=256, k=4, dc=28):
        super(transformer, self).__init__()
        r = 2*k+1
        self.conv1 = nn.Conv2d(in_chans*2, 256, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(256, 192, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(192, 128, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(128, 96, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(96, 64, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(64, dc, 3, 1, 1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(r)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        b = x.shape[0]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        x = self.pool(x)

        return x.view(b, 1, 1, 2, -1)

def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

if __name__ == '__main__':
    pass




