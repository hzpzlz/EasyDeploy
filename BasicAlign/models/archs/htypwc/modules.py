import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FlowEstimator(nn.Module):
    """docstring for FlowEstimator"""
    def __init__(self, ch_in):
        super(FlowEstimator, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1),
            conv(128,   128, 3, 1),
            conv(128,    96, 3, 1),
            conv(96,     64, 3, 1),
            conv(64,     32, 3, 1),
            predict_flow(32)
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


class Warp_Function(nn.Module):
    """docstring for Warp_Function"""
    def __init__(self):
        super(Warp_Function, self).__init__()

    def forward(self, img, flo):
        """
        img input dims: [B,C,H,W]
        flo input dims: [B,C,H,W]
        """
        B, C, H, W = tuple(int(i) for i in img.shape)
        img_flat = img.permute(0,2,3,1).reshape(-1, C)

        ## basic pos grid
        basic_grid = _coords_grid(B, H, W).float().cuda()
        pos_grid = torch.add(basic_grid, flo).permute(0,2,3,1).reshape(-1, 2)
        
        ## basic batch grid
        dim_batch = H * W
        batch_offsets = torch.arange(B) * dim_batch
        base_grid = batch_offsets.view(-1,1).repeat(1, dim_batch).view(-1)
        batch_grid = base_grid.type(torch.cuda.LongTensor)

        warped = _inter(img_flat, pos_grid, batch_grid, out_size=[H,W])
        warped_img = warped.reshape(-1, H, W, C).permute(0,3,1,2).float()

        return warped_img


#### basic grid for image
def _coords_grid(B, H, W):
    coords = torch.meshgrid(torch.arange(H), torch.arange(W))
    coords = torch.stack(coords[::-1], dim=0)
    return coords[None].repeat(B, 1, 1, 1)


#### bilinear interpolation
def _inter(img_flat, pos_grid, batch_grid, out_size):
    ## bilinear interpolation
    pos_floor = torch.floor(pos_grid)
    x0 = pos_floor[:, 0]
    y0 = pos_floor[:, 1]
    x1 = torch.add(x0, 1.0)
    y1 = torch.add(y0, 1.0)

    max_x = out_size[1] - 1.0
    max_y = out_size[0] - 1.0
    zero = 0.0

    x0 = torch.clamp(x0, zero, max_x).type(torch.cuda.LongTensor)
    x1 = torch.clamp(x1, zero, max_x).type(torch.cuda.LongTensor)
    y0 = torch.clamp(y0, zero, max_y).type(torch.cuda.LongTensor)
    y1 = torch.clamp(y1, zero, max_y).type(torch.cuda.LongTensor)
    
    bilinear_weights = torch.sub(pos_grid, pos_floor)
    xw = bilinear_weights[:, 0]
    yw = bilinear_weights[:, 1]
    
    ### Compute interpolation weights for 4 adjacent pixels
    ### expand to num_batch * height * width x 1 for broadcasting in add_n below
    wa = torch.mul((1.0 - xw), (1.0 - yw))[:, None]  ## top left pixel
    wb = torch.mul((1.0 - xw), yw)[:, None]          ## bottom left pixel
    wc = torch.mul(xw, (1.0 - yw))[:, None]          ## top right pixel
    wd = torch.mul(xw, yw)[:, None]                  ## bottom right pixel

    base_y0 = batch_grid + y0 * out_size[1]
    base_y1 = batch_grid + y1 * out_size[1]
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    Ia = torch.index_select(img_flat, dim=0, index=idx_a)
    Ib = torch.index_select(img_flat, dim=0, index=idx_b)
    Ic = torch.index_select(img_flat, dim=0, index=idx_c)
    Id = torch.index_select(img_flat, dim=0, index=idx_d)

    warped_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return warped_flat