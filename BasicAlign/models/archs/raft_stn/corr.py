import torch
import torch.nn.functional as F
from .utils import bilinear_sampler, coords_grid, bilinear_sampler_stn

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):  #下采样了3次 加上原图 一共有4种金字塔
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i] #batch*h1*w1, dim, h2, w2
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device) #构造出来的 3*3的范围

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i  #就是外面那个coords1 输入进来 外面网络学习的offset
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl   #构造出来的网格 + 学习到的坐标 作为最终的坐标 进行双线性插值  batch*h1*w1, 2*r+1, 2*r+1, 2

            corr = bilinear_sampler(corr, coords_lvl)  #出来的结果大小应该是 (batch*h1*w1, dim=1, 2*r+1, 2*r+1)
            corr = corr.view(batch, h1, w1, -1) #(batch, h1, w1, (2*r+1)**2)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)  #(batch, h1, w1, self.num_levels*(2*r+1)**2)
        return out.permute(0, 3, 1, 2).contiguous().float()  #(batch, self.num_levels*(2*r+1)**2, h1, w1) 得到的特征图

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)  #batch ht*wd ht*wd
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

class CorrBlock_stn:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock_stn.corr(fmap1, fmap2)

        #batch, h1, w1, dim, h2, w2 = corr.shape
        #batch, h1, w1, dim, h2, w2 = tuple(int(i) for i in corr.shape)
        #corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):  #下采样了3次 加上原图 一共有4种金字塔
            #corr = F.avg_pool2d(corr, 2, stride=2)

            new_size = tuple(int(i//2) for i in corr.shape[-2:])
            print(new_size, "------------------")
            #corr =  F.interpolate(corr, size=new_size, mode='nearest', align_corners=False)
            corr =  F.interpolate(corr, size=new_size, mode='nearest')
            #print(corr.shape)

            #H, W = tuple(int(i//2) for i in corr.shape[-2:])
            #corr = F.adaptive_avg_pool2d(corr, (H, W))
            print(corr.shape, "*******************")
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        #batch, h1, w1, _ = coords.shape
        batch, h1, w1, _ = tuple(int(i) for i in coords.shape)

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i] #batch*h1*w1, dim, h2, w2
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device) #构造出来的 3*3的范围

            #centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i  #就是外面那个coords1 输入进来 外面网络学习的offset
            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) * 2**(-i)  #就是外面那个coords1 输入进来 外面网络学习的offset
            #centroid_lvl = centroid_lvl.unsqueeze(1).unsqueeze(1)  #就是外面那个coords1 输入进来 外面网络学习的offset
            print(centroid_lvl.shape, "#########")
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl   #构造出来的网格 + 学习到的坐标 作为最终的坐标 进行双线性插值  batch*h1*w1, 2*r+1, 2*r+1, 2

            corr = bilinear_sampler_stn(corr, coords_lvl)  #出来的结果大小应该是 (batch*h1*w1, dim=1, 2*r+1, 2*r+1)
            corr = corr.view(batch, h1, w1, -1) #(batch, h1, w1, (2*r+1)**2)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)  #(batch, h1, w1, self.num_levels*(2*r+1)**2)
        return out.permute(0, 3, 1, 2).contiguous().float()  #(batch, self.num_levels*(2*r+1)**2, h1, w1) 得到的特征图

    @staticmethod
    def corr(fmap1, fmap2):
        #batch, dim, ht, wd = fmap1.shape
        batch, dim, ht, wd = tuple(int(i) for i in fmap1.shape)
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)  #batch ht*wd ht*wd
        #corr = corr.view(batch, ht, wd, 1, ht, wd)
        corr = corr.view(batch*ht*wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
