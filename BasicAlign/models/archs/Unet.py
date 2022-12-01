import torch
import torch.nn as nn
import torch.nn.functional as F
from .STN import STN


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
    def __init__(self, cpf=3, n_frame=5, out_nc=3):
        super(Unet, self).__init__()
        self.relu = nn.LeakyReLU(0.1,inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = nn.Conv2d(cpf*n_frame, 32, 3, 1, 1, bias=False)
        self.conv1_2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

        self.conv2_1 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)

        self.conv3_1 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

        self.conv4_1 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.conv4_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)

        self.conv5_1 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)

        self.uac6 = upsample_and_concat(256, 512)
        self.conv6_1 = nn.Conv2d(512, 256, 3, 1, 1, bias=False)
        self.conv6_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)

        self.uac7 = upsample_and_concat(128, 256)
        self.conv7_1 = nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.conv7_2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

        self.uac8 = upsample_and_concat(64, 128)
        self.conv8_1 = nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.conv8_2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)

        self.uac9 = upsample_and_concat(32, 64)
        self.conv9_1 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.conv9_2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

        self.conv10 = nn.Conv2d(32, out_nc, 3, 1, 1, bias=False)
        self.stn = STN.STN()

    def forward(self, x, test_mode=False):
        stn_out = self.stn(x)
        x = torch.cat(stn_out, 1)
        conv1 = self.relu(self.conv1_1(x))
        conv1 = self.relu(self.conv1_2(conv1))
        pool1 = self.pool(conv1)

        conv2 = self.relu(self.conv2_1(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        pool2 = self.pool(conv2)

        conv3 = self.relu(self.conv3_1(pool2))
        conv3 = self.relu(self.conv3_2(conv3))
        pool3 = self.pool(conv3)

        conv4 = self.relu(self.conv4_1(pool3))
        conv4 = self.relu(self.conv4_2(conv4))
        pool4 = self.pool(conv4)

        conv5 = self.relu(self.conv5_1(pool4))
        conv5 = self.relu(self.conv5_2(conv5))

        up6 = self.uac6(conv5, conv4)
        conv6 = self.relu(self.conv6_1(up6))
        conv6 = self.relu(self.conv6_2(conv6))

        up7 = self.uac7(conv6, conv3)
        conv7 = self.relu(self.conv7_1(up7))
        conv7 = self.relu(self.conv7_2(conv7))

        up8 = self.uac8(conv7, conv2)
        conv8 = self.relu(self.conv8_1(up8))
        conv8 = self.relu(self.conv8_2(conv8))
    
        up9 = self.uac9(conv8, conv1)
        conv9 = self.relu(self.conv9_1(up9))
        conv9 = self.relu(self.conv9_2(conv9))

        conv10 = self.conv10(conv9)

        #if test_mode:
        #    return stn_out
        #else:
        #    return conv10
        return conv10, stn_out
