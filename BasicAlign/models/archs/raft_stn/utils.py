import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from torch.autograd import Variable


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        #print(ht, wd)
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        #print(c[2], c[3], c[0], c[1], "****")
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)  #和grid的H W大小一样 通道数和img的一样

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, H, W):
    #coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    #coords = torch.stack(coords[::-1], dim=0).float()
    #grid = coords.unsqueeze(0).repeat(batch, 1, 1, 1)
    gridY = torch.arange(0, H, 1).view(1, -1, 1, 1).expand(batch, H, W, 1)
    gridX = torch.arange(0, W, 1).view(1, 1, -1, 1).expand(batch, H, W, 1)
    grid = torch.cat([gridX, gridY], dim=3).permute(0, 3, 1, 2).type(torch.FloatTensor) 
    return grid


def upflow8(flow, mode='bilinear'):
    #new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    new_size = tuple(int(i*8) for i in flow.shape[2:4])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def bilinear_sampler_stn(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    #batch, dim, H, W = img.shape
    batch, dim, H, W = tuple(int(i) for i in img.shape)
    xgrid, ygrid = coords.split([1,1], dim=-1)
    #xgrid = 2*xgrid/(W-1) - 1
    #ygrid = 2*ygrid/(H-1) - 1

    #grid = torch.cat([xgrid, ygrid], dim=-1)
    #img = F.grid_sample(img, grid, align_corners=True)  #和grid的H W大小一样 通道数和img的一样

    #grid = [xgrid, ygrid]
    #print(xgrid.shape, ygrid.shape, "---------------")
    xgrid = xgrid.reshape(-1)
    ygrid = ygrid.reshape(-1)

    out_size = tuple(int(i) for i in coords.shape[1:3])

    #img = img.permute([0, 2, 3, 1])
    img = warp(img, xgrid, ygrid, out_size)
    img = img.reshape(-1, out_size[0], out_size[1], dim).permute(0, 3, 1, 2)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def warp(im, x, y, out_size):
    #num_batch = im.size(0)
    #channels = im.size(1)
    #height = im.size(2)
    #width = im.size(3)
    num_batch, channels, height, width = tuple(int(i) for i in im.shape)

    x = x.float()
    y = y.float()

    out_height = out_size[0]
    out_width = out_size[1]
    zero = 0
    max_y = int(height - 1)
    max_x = int(width - 1)
    #max_y = height - 1
    #max_x = width - 1

    #x0 = x.floor().int()
    #y0 = y.floor().int()
    #x1 = x0 + 1
    #y1 = y0 + 1
    x0 = x.floor()
    y0 = y.floor()
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = torch.clamp(x0, zero, max_x).type(torch.cuda.LongTensor)
    x1 = torch.clamp(x1, zero, max_x).type(torch.cuda.LongTensor)
    y0 = torch.clamp(y0, zero, max_y).type(torch.cuda.LongTensor)
    y1 = torch.clamp(y1, zero, max_y).type(torch.cuda.LongTensor)
    dim2 = width
    dim1 = width * height

    base = _repeat(torch.arange(0, num_batch).int() * dim1, out_height * out_width)
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    im = im.permute(0, 2, 3, 1)
    im_flat = im.reshape(-1, channels).float()

    # print("interpolating... w/")
    # input(idx_a)

    #Ia = torch.index_select(im_flat, dim=0,
    #                        index=Variable(idx_a.type(torch.cuda.LongTensor)))  # .type(torch.cuda(3).FloatTensor)
    #Ib = torch.index_select(im_flat, dim=0,
    #                        index=Variable(idx_b.type(torch.cuda.LongTensor)))  # .type(torch.cuda(3).FloatTensor)
    #Ic = torch.index_select(im_flat, dim=0,
    #                        index=Variable(idx_c.type(torch.cuda.LongTensor)))  # .type(torch.cuda(3).FloatTensor)
    #Id = torch.index_select(im_flat, dim=0,
    #                        index=Variable(idx_d.type(torch.cuda.LongTensor)))  # .type(torch.cuda(3).FloatTensor)
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()
    wa = ((x1_f - x) * (y1_f - y))[:, None]
    wb = ((x1_f - x) * (y - y0_f))[:, None]
    wc = ((x - x0_f) * (y1_f - y))[:, None]
    wd = ((x - x0_f) * (y - y0_f))[:, None]

    A = wa * Ia.cuda().type(torch.cuda.FloatTensor)
    B = wb * Ib.cuda().type(torch.cuda.FloatTensor)
    C = wc * Ic.cuda().type(torch.cuda.FloatTensor)
    D = wd * Id.cuda().type(torch.cuda.FloatTensor)

    output = A + B + C + D
    return output

def _repeat(x, n_repeats):
    rep = torch.ones([1, n_repeats]).int()
    # There's some differnent between my implementation and original'
    # If something wrong, should change it back to original type
    x = torch.matmul(x.view(-1, 1), rep).cuda()
    return x.view(-1)
