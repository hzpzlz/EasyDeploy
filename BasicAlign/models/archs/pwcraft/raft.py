class CorrBlock:
    def __init__(self, num_levels=4, radius=4):
        super(CorrBlock, self).__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

    def forward(self, fmap1, fmap2, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        #batch, h1, w1, _ = coords.shape

        corr = self.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):  # 下采样了3次 加上原图 一共有4种金字塔
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]  # batch*h1*w1, dim, h2, w2
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)  # 构造出来的 3*3的范围

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i  # 就是外面那个coords1 输入进来 外面网络学习的offset
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl  # 构造出来的网格 + 学习到的坐标 作为最终的坐标 进行双线性插值  batch*h1*w1, 2*r+1, 2*r+1, 2

            corr = bilinear_sampler(corr, coords_lvl)  # 出来的结果大小应该是 (batch*h1*w1, dim=1, 2*r+1, 2*r+1)
            corr = corr.view(batch, h1, w1, -1)  # (batch, h1, w1, (2*r+1)**2)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)  # (batch, h1, w1, self.num_levels*(2*r+1)**2)
        return out.permute(0, 3, 1, 2).contiguous().float()  # (batch, self.num_levels*(2*r+1)**2, h1, w1) 得到的特征图

    def corr(self, fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)  # batch ht*wd ht*wd
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    #img = F.grid_sample(img, grid, align_corners=True)  #和grid的H W大小一样 通道数和img的一样
    img = bilinear_interpolate_torch_2D(img, grid, align_corners=True)  #和grid的H W大小一样 通道数和img的一样

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2*corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1) #这里的80

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)  #输出是82 通道的 80+2

class SmallUpdateBlock(nn.Module):
    def __init__(self, corr_levels, corr_radius, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(corr_levels, corr_radius)  # 上面corr后的结果
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)#两层conv预测光流变化

    def forward(self, net, corr, flow):
        motion_features = self.encoder(flow, corr)
        #inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, motion_features)
        delta_flow = self.flow_head(net)

        return net, delta_flow