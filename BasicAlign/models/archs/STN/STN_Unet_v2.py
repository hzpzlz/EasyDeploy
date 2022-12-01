import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels, fc_in=128, kernel_size=3, use_dropout=False):
        super(SpatialTransformer, self).__init__()
        #self._h, self._w = spatial_dims
        self._in_ch = in_channels
        self._ksize = kernel_size
        self.dropout = use_dropout
        self.fc_in = fc_in

        # localization net
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False) # size : [1x3x32x32]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, self.fc_in, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.pool5 = nn.AdaptiveAvgPool2d(4)

        self.fc1 = nn.Linear(self.fc_in*4*4, 512)
        self.fc2 = nn.Linear(512, 6)

    def forward(self, x, shape):
        """
        Forward pass of the STN module.
        x -> input feature map
        """
        batch_images = x
        self._h, self._w = shape
        #x = F.relu(self.conv1(x.detach()))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        #x = F.max_pool2d(x, 2)
        x = self.pool5(x)
        #print("Pre view size:{}".format(x.size()))
        x = x.view(-1, self.fc_in*4*4)
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


class Unet(nn.Module):
    def __init__(self, if_stn=True, in_nc=3, out_nc=3):
        super(Unet, self).__init__()
        self.relu = nn.LeakyReLU(0.1,inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = nn.Conv2d(in_nc, 32, 3, 1, 1, bias=False)
        self.conv1_2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        if if_stn:
            self.stn1 = SpatialTransformer(32, 32)
        else:
            self.stn1 = None

        self.conv2_1 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        if if_stn:
            self.stn2 = SpatialTransformer(64, 64)
        else:
            self.stn2 = None

        self.conv3_1 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        if if_stn:
            self.stn3 = SpatialTransformer(128, 128)
        else:
            self.stn3 = None

        self.conv4_1 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.conv4_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        if if_stn:
            self.stn4 = SpatialTransformer(256, 256)
        else:
            self.stn4 = None

        self.conv5_1 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        if if_stn:
            self.stn5 = SpatialTransformer(512, 512)
        else:
            self.stn5 = None

        self.uac6 = upsample_and_concat(256, 512)
        if if_stn:
            self.stn6 = SpatialTransformer(256)
        else:
            self.stn6 = None
        self.conv6_0 = nn.Conv2d(512, 256, 3, 1, 1, bias=False)
        self.conv6_1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv6_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)

        self.uac7 = upsample_and_concat(128, 256)
        if if_stn:
            self.stn7 = SpatialTransformer(128)
        else:
            self.stn7 = None
        self.conv7_0 = nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.conv7_1 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv7_2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

        self.uac8 = upsample_and_concat(64, 128)
        if if_stn:
            self.stn8 = SpatialTransformer(64)
        else:
            self.stn8 = None
        self.conv8_0 = nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.conv8_1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv8_2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)

        self.uac9 = upsample_and_concat(32, 64)
        if if_stn:
            self.stn9 = SpatialTransformer(32)
        else:
            self.stn9 = None
        self.conv9_0 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.conv9_1 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv9_2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

        self.conv10 = nn.Conv2d(32, out_nc, 3, 1, 1, bias=False)

    def forward(self, x, if_stn=True):
        #f1, f2, f3, f4, f5 = torch.split(x, 3, 1)
        conv1 = self.relu(self.conv1_1(x))
        conv1 = self.relu(self.conv1_2(conv1))
        if self.stn1 is not None:
            conv1 = self.stn1(conv1, conv1.shape[-2:])
        pool1 = self.pool(conv1)

        conv2 = self.relu(self.conv2_1(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        if self.stn2 is not None:
            conv2 = self.stn2(conv2, conv2.shape[-2:])
        pool2 = self.pool(conv2)

        conv3 = self.relu(self.conv3_1(pool2))
        conv3 = self.relu(self.conv3_2(conv3))
        if self.stn3 is not None:
            conv3 = self.stn3(conv3, conv3.shape[-2:])
        pool3 = self.pool(conv3)

        conv4 = self.relu(self.conv4_1(pool3))
        conv4 = self.relu(self.conv4_2(conv4))
        if self.stn4 is not None:
            conv4 = self.stn4(conv4, conv4.shape[-2:])
        pool4 = self.pool(conv4)

        conv5 = self.relu(self.conv5_1(pool4))
        conv5 = self.relu(self.conv5_2(conv5))
        if self.stn5 is not None:
            conv5 = self.stn5(conv5, conv5.shape[-2:])

        up6 = self.uac6(conv5, conv4)
        up6 = self.relu(self.conv6_0(up6))
        conv6 = self.relu(self.conv6_1(up6))
        conv6 = self.relu(self.conv6_2(conv6))
        if self.stn6 is not None:
            conv6 = self.stn6(conv6, conv6.shape[-2:])

        up7 = self.uac7(conv6, conv3)
        up7 = self.relu(self.conv7_0(up7))
        conv7 = self.relu(self.conv7_1(up7))
        conv7 = self.relu(self.conv7_2(conv7))
        if self.stn7 is not None:
            conv7 = self.stn7(conv7, conv7.shape[-2:])

        up8 = self.uac8(conv7, conv2)
        up8 = self.relu(self.conv8_0(up8))
        conv8 = self.relu(self.conv8_1(up8))
        conv8 = self.relu(self.conv8_2(conv8))
        if self.stn8 is not None:
            conv8 = self.stn8(conv8, conv8.shape[-2:])
    
        up9 = self.uac9(conv8, conv1)
        up9 = self.relu(self.conv9_0(up9))
        conv9 = self.relu(self.conv9_1(up9))
        conv9 = self.relu(self.conv9_2(conv9))
        if self.stn9 is not None:
            conv9 = self.stn9(conv9, conv9.shape[-2:])

        conv10 = self.conv10(conv9)

        return [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10]

class STN_Unet_v2(nn.Module):
    def __init__(self, cpf=3, n_frame=5, out_nc=3):
        super(STN_Unet_v2, self).__init__()
        self.unet1 = Unet()
        self.unet2 = Unet()
        self.unet3 = Unet(if_stn=False)
        self.unet4 = Unet()
        self.unet5 = Unet()

    def forward(self, x):
        f1, f2, f3, f4, f5 = torch.split(x, 3, 1)
        f1 = self.unet1(f1)
        f2 = self.unet2(f2)
        f3_ = self.unet3(f3)
        f4 = self.unet4(f4)
        f5 = self.unet5(f5)
        #print(f1.shape, f2.shape, f3.shape, f4.shape, f5.shape)

        return [f1[-1], f2[-1], f3_[-1], f4[-1], f5[-1]], [f1, f2, f3_, f4, f5], [f3_[-1], f3]



