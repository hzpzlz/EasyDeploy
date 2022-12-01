import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
import math


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


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def bilinear_sampler_stn(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    batch, dim, H, W = img.shape
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    #grid = torch.cat([xgrid, ygrid], dim=-1)
    #img = F.grid_sample(img, grid, align_corners=True)  #和grid的H W大小一样 通道数和img的一样

    #grid = [xgrid, ygrid]
    xgrid = xgrid.reshape(-1)
    ygrid = ygrid.reshape(-1)

    out_size = coords.shape[1:3]

    img = img.permute([0, 2, 3, 1])
    img = warp(img, xgrid, ygrid, out_size)
    img = img.reshape(-1, dim, out_size[0], out_size[1])

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def _repeat(x, n_repeats):
    rep = torch._cast_Long(torch.transpose(torch.ones([n_repeats, ]).unsqueeze(1), 1, 0))
    #rep = torch.LongTensor(torch.transpose(torch.ones([n_repeats, ]).unsqueeze(1), 1, 0))
    x = torch.matmul(x.view(-1, 1), rep)
    return x.view(-1)

def warp(im, x, y, out_size):  #可以用来warp  im +x_offset +y_offset +out_size
    num_batch, height, width, channels = im.size()  # to be sure the input dims is NHWC
    x = torch._cast_Float(x).cuda()
    y = torch._cast_Float(y).cuda()
    height_f = torch._cast_Float(torch.Tensor([height]))[0].cuda()
    width_f = torch._cast_Float(torch.Tensor([width]))[0].cuda()
    #x = x.cuda()
    #y = y.cuda()
    #height_f = torch.Tensor([height])[0].cuda()
    #width_f = torch.Tensor([width])[0].cuda()
    out_height = out_size[0]
    out_width = out_size[1]
    zero = torch.zeros([], dtype=torch.int32).cuda()
    max_y = torch._cast_Long(torch.Tensor([height - 1]))[0].cuda()
    max_x = torch._cast_Long(torch.Tensor([width - 1]))[0].cuda()
    #max_y = torch.Tensor([height - 1])[0].cuda()
    #max_x = torch.Tensor([width - 1])[0].cuda()

    # scale indices from [-1, 1] to [0, width/height]
    x = (x + 1.0) * width_f / 2.0
    y = (y + 1.0) * height_f / 2.0

    # do sampling
    x0 = torch._cast_Long(torch.floor(x)).cuda()
    #x0 = torch.floor(x).cuda()
    x1 = x0 + 1
    y0 = torch._cast_Long(torch.floor(y)).cuda()
    #y0 = torch.floor(y).cuda()
    y1 = y0 + 1

    x0 = torch.clamp(x0, zero, max_x)
    x1 = torch.clamp(x1, zero, max_x)
    y0 = torch.clamp(y0, zero, max_y)
    y1 = torch.clamp(y1, zero, max_y)
    dim2 = width
    dim1 = width * height
    base = _repeat(torch.arange(num_batch) * dim1, out_height * out_width).cuda()
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to look up pixels in the flate images
    # and restore channels dim
    im_flat = im.contiguous().view(-1, channels)
    im_flat = torch._cast_Float(im_flat)
    #im_flat = im_flat
    Ia = im_flat[idx_a].cuda()  # as in tf, the default dim is row first
    Ib = im_flat[idx_b].cuda()
    Ic = im_flat[idx_c].cuda()
    Id = im_flat[idx_d].cuda()

    # calculate interpolated values
    x0_f = torch._cast_Float(x0).cuda()
    x1_f = torch._cast_Float(x1).cuda()
    y0_f = torch._cast_Float(y0).cuda()
    y1_f = torch._cast_Float(y1).cuda()
    #x0_f = x0.cuda()
    #x1_f = x1.cuda()
    #y0_f = y0.cuda()
    #y1_f = y1.cuda()
    wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
    wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
    wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
    wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def get_4_pts(theta, mesh_num, batch_size):
    grid_h = grid_w = mesh_num
    do_crop_rate = 0.8
    pts1_ = []
    pts2_ = []
    pts = []
    h = 2.0 / grid_h
    w = 2.0 / grid_w
    tot = 0
    for i in range(grid_h + 1):
        pts.append([])
        for j in range(grid_w + 1):
            hh = i * h - 1
            ww = j * w - 1
            p = torch._cast_Float(torch.Tensor([ww, hh]).view(2)).cuda()
            temp = theta[:, tot * 2: tot * 2 + 2]
            tot += 1
            p = (p + temp).view([batch_size, 1, 2])
            p = torch.clamp(p, -1. / do_crop_rate, 1. / do_crop_rate)
            pts[i].append(p.view([batch_size, 2, 1]))
            pts2_.append(p)

    for i in range(grid_h):
        for j in range(grid_w):
            g = torch.cat([pts[i][j], pts[i][j + 1], pts[i + 1][j], pts[i + 1][j + 1]], dim = 2)
            pts1_.append(g.view([batch_size, 1, 8]))

    pts1 = torch.cat(pts1_, 1).view([batch_size, grid_h, grid_w, 8])
    pts2 = torch.cat(pts2_, 1).view([batch_size, grid_h + 1, grid_w + 1, 2])

    return pts1, pts2

def get_Hs(theta, mesh_num):
    #_, _, height, width = U.shape
    #print(theta.shape, "ttttttttttttttttttttt")
    grid_h = grid_w = mesh_num
    num_batch = theta.size()[0]
    h = 2.0 / grid_h
    w = 2.0 / grid_w
    Hs = []
    for i in range(grid_h):
        for j in range(grid_w):
            hh = i * h - 1
            ww = j * w - 1
            ori = torch._cast_Float(torch.Tensor([ww, hh, ww + w, hh, ww, hh + h, ww + w, hh + h])). \
                view([1, 8]).repeat([num_batch, 1]).cuda()
            id = i * (grid_w + 1) + grid_w
            tar = torch.cat([theta[:, i:i + 1, j:j + 1, :],
                            theta[:, i:i + 1, j + 1:j + 2, :],
                            theta[:, i + 1:i + 2, j:j + 1, :],
                            theta[:, i + 1:i + 2, j + 1:j + 2, :]], dim=1)
            tar = tar.view([num_batch, 8])
            Hs.append(get_H(ori, tar).view([num_batch, 1, 9]))

    Hs = torch.cat(Hs, dim=1).view([num_batch, grid_h, grid_w, 9])
    return Hs

def pinv(A):
    A = A.cpu() + torch.eye(8) * 1e-4
    #A = A.cuda() + torch.eye(8).cuda() * 1e-4
    #print(A.shape, "AAAAAAAAAAAAAAAAAA")
    return torch.inverse(A).cuda()

def get_H(ori, tar):  #计算H矩阵 通过8个方程求解H，A*H=b H=A' * b  A'表示A的转置
    #ttt = time.time()
    num_batch = ori.size()[0]
    one = torch.ones([num_batch, 1]).cuda()
    zero = torch.zeros([num_batch, 1]).cuda()
    x = [ori[:, 0:1], ori[:, 2:3], ori[:, 4:5], ori[:, 6:7]] #取出对应的x y坐标
    y = [ori[:, 1:2], ori[:, 3:4], ori[:, 5:6], ori[:, 7:8]]
    u = [tar[:, 0:1], tar[:, 2:3], tar[:, 4:5], tar[:, 6:7]]
    v = [tar[:, 1:2], tar[:, 3:4], tar[:, 5:6], tar[:, 7:8]]

    A_ = []
    A_.extend([x[0], y[0], one, zero, zero, zero, -x[0] * u[0], -y[0] * u[0]])
    A_.extend([x[1], y[1], one, zero, zero, zero, -x[1] * u[1], -y[1] * u[1]])
    A_.extend([x[2], y[2], one, zero, zero, zero, -x[2] * u[2], -y[2] * u[2]])
    A_.extend([x[3], y[3], one, zero, zero, zero, -x[3] * u[3], -y[3] * u[3]])
    A_.extend([zero, zero, zero, x[0], y[0], one, -x[0] * v[0], -y[0] * v[0]])
    A_.extend([zero, zero, zero, x[1], y[1], one, -x[1] * v[1], -y[1] * v[1]])
    A_.extend([zero, zero, zero, x[2], y[2], one, -x[2] * v[2], -y[2] * v[2]])
    A_.extend([zero, zero, zero, x[3], y[3], one, -x[3] * v[3], -y[3] * v[3]])
    A = torch.cat(A_, dim=1).view(num_batch, 8, 8)
    b_ = [u[0], u[1], u[2], u[3], v[0], v[1], v[2], v[3]]
    b = torch.cat(b_, dim=1).view([num_batch, 8, 1])

    ans = torch.cat([torch.matmul(pinv(A), b).view([num_batch, 8]), torch.ones([num_batch, 1]).cuda()],
                    dim=1)
    #iii = time.time()
    #print("hhhhhhhhhh", iii-ttt)
    return ans

def meshgrid2(height, width, sh, eh, sw, ew):
    hn = eh - sh + 1
    wn = ew - sw + 1
    #print(hn, wn)  75 100 height//mesh_num weight//mesh_num

    #取出坐标中某一块的坐标值
    x_t = torch.matmul(torch.ones([hn, 1]).cuda(),
                           torch.transpose(torch.linspace(-1.0, 1.0, width)[sw:sw + wn].unsqueeze(1), 1, 0).cuda())
    y_t = torch.matmul(torch.linspace(-1.0, 1.0, height)[sh:sh + hn].unsqueeze(1).cuda(),
                           torch.ones([1, wn]).cuda())
    #x_t = torch.matmul(torch.ones([hn, 1]).cuda(),
    #                   torch.transpose(torch.linspace(0, width, width)[sw:sw + wn].unsqueeze(1), 1, 0).cuda())
    #y_t = torch.matmul(torch.linspace(0, height, height)[sh:sh + hn].unsqueeze(1).cuda(),
    #                   torch.ones([1, wn]).cuda())

    x_t_flat = x_t.view(1, -1)
    y_t_flat = y_t.view(1, -1)

    ones = torch.ones_like(x_t_flat)
    grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
    return grid

def create_grid(theta, input_dim, mesh_num):
    #print(input_dim.shape, "uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
    num_batch, num_channels, height, width = input_dim.shape

    grid_h = grid_w = mesh_num
    #input_dim = input_dim.permute([0, 2, 3, 1])
    #num_batch = input_dim.size()[0]
    #num_channels = input_dim.size()[3]
    #theta = torch._cast_Float(theta)
    Hs = get_Hs(theta, mesh_num)
    gh = int(math.floor(height / grid_h))
    gw = int(math.floor(width / grid_w))
    x_ = []
    y_ = []

    for i in range(grid_h):
        row_x_ = []
        row_y_ = []
        for j in range(grid_w):
            H = Hs[:, i:i + 1, j:j + 1, :].view(num_batch, 3, 3)
            sh = i * gh
            eh = (i + 1) * gh - 1
            sw = j * gw
            ew = (j + 1) * gw - 1
            if (i == grid_h - 1):
                eh = height - 1
            if (j == grid_w - 1):
                ew = width - 1
            grid = meshgrid2(height, width, sh, eh, sw, ew)
            grid = grid.unsqueeze(0)
            grid = grid.repeat([num_batch, 1, 1])

            T_g = torch.matmul(H, grid)
            x_s = T_g[:, 0:1, :]
            y_s = T_g[:, 1:2, :]
            z_s = T_g[:, 2:3, :]

            z_s_flat = z_s.contiguous().view(-1)
            t_1 = torch.ones(z_s_flat.size()).cuda()
            t_0 = torch.zeros(z_s_flat.size()).cuda()

            sign_z_flat = torch.where(z_s_flat >= 0, t_1, t_0) * 2 - 1
            z_s_flat = z_s.contiguous().view(-1) + sign_z_flat * 1e-8
            x_s_flat = x_s.contiguous().view(-1) / z_s_flat
            y_s_flat = y_s.contiguous().view(-1) / z_s_flat

            x_s = x_s_flat.view([num_batch, eh - sh + 1, ew - sw + 1])
            y_s = y_s_flat.view([num_batch, eh - sh + 1, ew - sw + 1])
            row_x_.append(x_s)
            row_y_.append(y_s)
        row_x = torch.cat(row_x_, dim=2)
        row_y = torch.cat(row_y_, dim=2)
        x_.append(row_x)
        y_.append(row_y)

    x = torch.cat(x_, dim=1).view([num_batch, height, width, 1])
    y = torch.cat(y_, dim=1).view([num_batch, height, width, 1])
    #grid = torch.cat([x ,y], 3).permute(0, 3, 1, 2)
    return x, y
