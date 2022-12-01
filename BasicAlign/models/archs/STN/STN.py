import torch
import torch.nn as nn
import torch.nn.functional as F


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

class SpatialTransformer(nn.Module):
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
    def __init__(self, in_channels, kernel_size=3, use_dropout=False):
        super(SpatialTransformer, self).__init__()
        #self._h, self._w = spatial_dims
        self._in_ch = in_channels
        self._ksize = kernel_size
        self.dropout = use_dropout

        # localization net
        self.conv1 = nn.Conv2d(in_channels*2, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False) # size : [1x3x32x32]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.pool5 = nn.AdaptiveAvgPool2d(4)

        self.fc1 = nn.Linear(256*4*4, 1024)
        self.fc2 = nn.Linear(1024, 6)


    def forward(self, ref, base):
        """
        Forward pass of the STN module.
        x -> input feature map
        """
        batch_images = ref
        self._h, self._w = ref.shape[2:]
        #x = F.relu(self.conv1(x.detach()))
        x = F.relu(self.conv1(torch.cat([ref, base], 1)))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv4(x))
        #x = F.max_pool2d(x, 2)
        x = self.pool5(x)
        #print("Pre view size:{}".format(x.size()))
        x = x.view(-1, 256*4*4)
        if self.dropout:
            x = F.dropout(self.fc1(x), p=0.5)
            x = F.dropout(self.fc2(x), p=0.5)
        else:
            x = self.fc1(x)
            x = self.fc2(x) # params [Nx6]

        x = x.view(-1, 2, 3) # change it to the 2x3 matrix
        #print(x.size())
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        #print(affine_grid_points.size(0), batch_images.size(0), "************************")
        assert(affine_grid_points.size(0) == batch_images.size(0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(batch_images, affine_grid_points)
        #print("rois found to be of size:{}".format(rois.size()))
        return rois

class STN(nn.Module):
    def __init__(self, cpf=3, n_frame=5, out_nc=3):
        super(STN, self).__init__()
        self.relu = nn.LeakyReLU(0.1,inplace=True)
        self.cpf = cpf
        self.n_frame = n_frame
        self.fc_in = 512

        self.conv1_1 = nn.Conv2d(n_frame*cpf, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_3 = nn.Conv2d(512, self.fc_in, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool = nn.AdaptiveAvgPool2d(4)

        self.fc1 = nn.Linear(self.fc_in*4*4, 1024)
        self.fc2 = nn.Linear(1024, 6*(n_frame-1))

    def forward(self, x):
        f1, f2, f3, f4, f5 = torch.split(x, self.cpf, 1)
        self.batch, self._in_ch, self._h, self._w = f3.shape
        #print(f3)
        x = self.relu(self.conv1_1(x))
        #print(x)
        x = self.relu(self.conv1_2(x))
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.relu(self.conv2_3(x))
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.relu(self.conv4_1(x))
        #print(x)
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.relu(self.conv5_1(x))
        #print(x)
        x = self.relu(self.conv5_2(x))
        x = self.conv5_3(x)
        #print(x, "1111111111111111111111111111")
        x = self.pool(x)
        #print(x, "2222222222222222222222222222")
        x = x.view(-1, self.fc_in*4*4)
        x = self.fc1(x)
        x = self.fc2(x)

        x1, x2, x4, x5 = torch.split(x, 6, 1)
        x1 = x1.view(-1, 2, 3)
        x2 = x2.view(-1, 2, 3)
        #x3 = x3.view(-1, 2, 3)
        x4 = x4.view(-1, 2, 3)
        x5 = x5.view(-1, 2, 3)
        #print(x1)

        grid1 = F.affine_grid(x1, torch.Size((self.batch, self._in_ch, self._h, self._w)))
        grid2 = F.affine_grid(x2, torch.Size((self.batch, self._in_ch, self._h, self._w)))
        #grid3 = F.affine_grid(x3, torch.Size((self.batch, self._in_ch, self._h, self._w)))
        grid4 = F.affine_grid(x4, torch.Size((self.batch, self._in_ch, self._h, self._w)))
        grid5 = F.affine_grid(x5, torch.Size((self.batch, self._in_ch, self._h, self._w)))
        #print(affine_grid_points.size(0), batch_images.size(0), "************************")
        assert(grid1.size(0) == self.batch), "The batch sizes of the input images must be same as the generated grid."
        #print(grid1)
        f1_w = F.grid_sample(f1, grid1)
        f2_w = F.grid_sample(f2, grid2)
        #f3_w = F.grid_sample(f3, grid3)
        f4_w = F.grid_sample(f4, grid4)
        f5_w = F.grid_sample(f5, grid5)
        #print(f1_w)

        return [f1_w, f2_w, f3, f4_w, f5_w]
