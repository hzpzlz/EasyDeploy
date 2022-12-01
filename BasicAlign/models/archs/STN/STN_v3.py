import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import transformer


class upsample_and_concat(nn.Module):
    def __init__(self, out_nc, in_nc):
        super(upsample_and_concat, self).__init__()
        #self.upsample = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv = nn.Conv2d(in_nc, out_nc, 3, 1, 1, bias=False)

    def forward(self, x1, x2):
        #x1_up = self.upsample(x1)
        x1_up = F.interpolate(x1, x2.shape[-2:], mode='nearest')
        x1_up = self.conv(x1_up)
        return torch.cat([x1_up, x2], 1)

class ST(nn.Module):
    """
    Implements a spatial transformer
    as proposed in the Jaderberg paper.
    Comprises of 3 parts:
    1. Localization Net
    2. A grid generator
    3. A roi pooled module.
    The current implementation uses a very small convolutional net with
    2 convolutional layers and 2 fully connected layers. Backends
    can be swapped in favor of VGG, ResNets etc. TTMV
    Returns:
    A roi feature map with the same input spatial dimension as the input feature map.
    """
    def __init__(self, in_channels, fc_in=512, kernel_size=3):
        super(ST, self).__init__()
        #self._h, self._w = spatial_dims
        self._in_ch = in_channels
        self._ksize = kernel_size
        #self.dropout = use_dropout
        self.fc_in = fc_in

        # localization net
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False) # size : [1x3x32x32]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(256, self.fc_in, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.pool5 = nn.AdaptiveAvgPool2d(4)

        self.fc1 = nn.Linear(self.fc_in*4*4, 1024)
        self.fc2 = nn.Linear(1024, 8)


    def forward(self, base, x):
        """
        Forward pass of the STN module.
        x -> input feature map
        """
        #batch_images = x
        self._batch, _, self._h, self._w = base.shape
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = x.view(-1, self.fc_in*4*4)
        x = self.fc1(x)
        x = self.fc2(x) # params [Nx6]
        x_cat = torch.ones(self._batch, 1).cuda()

        grid = torch.cat([x, x_cat], 1) # change it to the 2x3 matrix
        #affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        #assert(affine_grid_points.size(0) == base.size(0)), "The batch sizes of the input images must be same as the generated grid."
        #rois = F.grid_sample(batch_images, affine_grid_points)
        rois = transformer(base, grid, (self._h, self._w))
        return rois.permute(0, 3, 1, 2)

class conv(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(conv, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x

class STN(nn.Module):
    def __init__(self, cpf=3, out_nc=3):
        super(STN, self).__init__()
        self.relu = nn.LeakyReLU(0.1,inplace=True)
        self.cpf = cpf

        self.ST1 = ST(cpf)
        self.conv1 = conv(cpf+3, 32)
        self.ST2 = ST(32)
        self.conv2 = conv(32, 64)
        self.ST3 = ST(64)
        self.conv3 = conv(64, 32)

        self.conv_out = nn.Conv2d(32, out_nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.stn_out = ST(out_nc)

    def forward(self, x, gt):
        base = x
        x = self.ST1(base, x)
        x = self.conv1(torch.cat([x, gt], 1))
        x = self.ST2(base, x)
        x = self.conv1(torch.cat([x, gt], 1))
        x = self.conv2(x)
        x = self.ST3(base, x)
        x = self.conv1(torch.cat([x, gt], 1))
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.conv_out(x)
        x = self.stn_out(base, x)

        return x

class STN_v3(nn.Module):
    def __init__(self, cpf=3, n_frame=5, out_nc=3):
        super(STN_v3, self).__init__()
        self.stn1 = STN()
        self.stn2 = STN()
        self.stn4 = STN()
        self.stn5 = STN()

    def forward(self, x):
        f1, f2, f3, f4, f5 = torch.split(x, 3, 1)
        f1 = self.stn1(f1, f3)
        f2 = self.stn2(f2, f3)
        f4 = self.stn4(f4, f3)
        f5 = self.stn5(f5, f3)

        return [f1, f2, f3, f4, f5]
