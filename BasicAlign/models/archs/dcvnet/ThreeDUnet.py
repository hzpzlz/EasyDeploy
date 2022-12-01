import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):

    def __init__(self, in_chans, out_chans, rate=1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch5_avg = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(out_chans)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_chans * 5, out_chans, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = self.branch5_avg(x)
        global_feature = self.branch5_relu(self.branch5_bn(self.branch5_conv(global_feature)))
        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)

        return result.reshape(b, c, t, h, w)

class ThreeDUnet(nn.Module):
    def __init__(self, out_dim=7):
        super(ThreeDUnet, self).__init__()
        self.conv1_1 = nn.Conv3d(28, 49, kernel_size=(7, 3, 3), stride=1, padding=(3, 1, 1), groups=7)
        self.conv1_2 = nn.Conv3d(49, 49, kernel_size=(7, 3, 3), stride=1, padding=(3, 1, 1), groups=7)
        self.conv1_3 = nn.Conv3d(49, 48, kernel_size=(7, 3, 3), stride=1, padding=(3, 1, 1))

        self.conv2_1 = nn.Conv3d(48, 96, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))
        self.conv2_2 = nn.Conv3d(96, 96, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.conv3_1 = nn.Conv3d(96, 96, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))
        self.conv3_2 = nn.Conv3d(96, 96, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3_3 = nn.Conv3d(96, 96, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.aspp = ASPP(96, 96)

        self.conv4_1 = nn.Conv3d(96*2, 96, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv4_2 = nn.Conv3d(96, 96, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.conv5_1 = nn.Conv3d(48*2, 48, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv5_2 = nn.Conv3d(48, 48, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.out_conv = nn.Conv3d(48, out_dim, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.relu(self.conv1_2(self.relu(self.conv1_1(x))))
        print(x.shape, "xxxxxxxxxx")
        x = x1 = self.relu(self.conv1_3(x))
        print(x.shape, "222222222")
        x = x2 = self.relu(self.conv2_2(self.relu(self.conv2_1(x))))
        x = self.relu(self.conv3_3(self.relu(self.conv3_2(self.relu(self.conv3_1(x))))))
        print(x.shape, "xxxxxxxxxx")
        x = self.aspp(x)
        print(x.shape, "xxxxxxxxxx")
        x = F.interpolate(x, (x.shape[-2:][0]*2, x.shape[-2:][1]*2), None, 'bilinear', True)
        x = torch.cat([x, x1], dim=1)
        x = self.relu(self.conv4_2(self.relu(self.conv4_1(x))))
        x = F.interpolate(x, x.shape[-2:]*2, None, 'bilinear', True)
        x = torch.cat([x, x2], dim=1)
        x = self.relu(self.conv5_2(self.relu(self.conv5_1(x))))

        out = self.out_conv(x)
        return out

if __name__ == '__main__':
    x = torch.rand(1, 28, 81, 64, 64)
    print(x.shape)
    unet = ThreeDUnet()
    out = unet(x)
    print(out.shape)
    #out = unet(x.permute(0, 2, 1, 3, 4))
    #print(out.permute(0, 2, 1, 3, 4).shape)




