import torch
import torch.nn as nn
import torch.nn.functional as F
from .archs.maskflownet.MaskFlownet import Upsample, Downsample

from torch.autograd import Variable
import numpy as np
from math import exp
from models.archs.vgg_arch import VGGFeatureExtractor

def type_trans(window,img):
    if img.is_cuda:
        window = window.cuda(img.get_device())
    return window.type_as(img)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):

    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
    # print(mu1.shape,mu2.shape)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mcs_map  = (2.0 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    # print(ssim_map.shape)
    if size_average:
        return ssim_map.mean(), mcs_map.mean()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        window = create_window(self.window_size,channel)
        window = type_trans(window,img1)
        ssim_map, mcs_map =_ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return ssim_map

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue
 
class L2(nn.Module):
    def __init__(self):
         super(L2, self).__init__()
         
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        filterx = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0. , 3.]])
        self.fx = filterx.expand(1,3,3,3).cuda()

        filtery = torch.tensor([[-3., -10, -3.], [0., 0., 0.], [3., 10. , 3.]])
        self.fy = filtery.expand(1,3,3,3).cuda()

    def forward(self, x, y):
        schxx = F.conv2d(x, self.fx, stride=1, padding=1)
        schxy = F.conv2d(x, self.fy, stride=1, padding=1)
        gradx = torch.sqrt(torch.pow(schxx, 2) + torch.pow(schxy, 2) + 1e-6)
        
        schyx = F.conv2d(y, self.fx, stride=1, padding=1)
        schyy = F.conv2d(y, self.fy, stride=1, padding=1)
        grady = torch.sqrt(torch.pow(schyx, 2) + torch.pow(schyy, 2) + 1e-6)
        
        loss = F.l1_loss(gradx, grady)
        return loss

class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=False):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img


class FilterHigh(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, include_pad=True, normalize=True, gaussian=False):
        super(FilterHigh, self).__init__()
        self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, include_pad=include_pad,
                                    gaussian=gaussian)
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):
        if self.recursions > 1:
            for i in range(self.recursions - 1):
                img = self.filter_low(img)
        img = img - self.filter_low(img)
        if self.normalize:
            return 0.5 + img * 0.5
        else:
            return img

class FSLoss(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, gaussian=False):
        super(FSLoss, self).__init__()
        self.filter = FilterHigh(recursions=recursions, stride=stride, kernel_size=kernel_size, include_pad=False,
                                     gaussian=gaussian)
    def forward(self, x, y):
        x_ = self.filter(x)
        y_ = self.filter(y)
        loss = F.l1_loss(x_, y_)
        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        #loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

class Flow_Loss(nn.Module):
    def __init__(self):
        super(Flow_Loss, self).__init__()

    def forward(self, flow_preds, flow_gt, valid,  max_flow=400):
        """ Loss function defined over sequence of flow predictions """

        # exlude invalid pixels and extremely large diplacements 排除无效像素和超大位移
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < max_flow)

        i_loss = (flow_preds - flow_gt).abs()
        flow_loss = (valid[:, None] * i_loss).mean()

        epe = torch.sum((flow_preds - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return flow_loss, metrics
    

class Sequence_Loss(nn.Module):
    def __init__(self, gamma):
        super(Sequence_Loss, self).__init__()
        self.gamma = gamma

    def forward(self, flow_preds, flow_gt, valid,  max_flow=400):
        """ Loss function defined over sequence of flow predictions """
        n_predictions = len(flow_preds)    
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements 排除无效像素和超大位移
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma**(n_predictions - i - 1) #越到后面权重越小
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return flow_loss, metrics

class Sequence_Warp(nn.Module):
    def __init__(self, gamma, loss_type='l1'):
        super(Sequence_Warp, self).__init__()
        self.gamma = gamma
        if loss_type=='l1':
            self.loss = nn.L1Loss()
        elif loss_type=='l2':
            self.loss = nn.MSELoss()

    def forward(self, img_preds, img_gt):
        """ Loss function defined over sequence of flow predictions """
        n_predictions = len(img_preds)    
        img_loss = 0.0


        for i in range(n_predictions):
            i_weight = self.gamma**(n_predictions - i - 1) #越到后面权重越da
            i_loss = self.loss(img_preds[i], img_gt)
            img_loss += i_weight * i_loss

        return img_loss

def downflow(flow, scale, mode='bilinear'):
    new_size = (flow.shape[2] // scale, flow.shape[3] // scale)
    return 0.05 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

class PWC_Loss(nn.Module):
    def __init__(self, output_level=4, l_weight= 0.32, norm= 'L2'):
        super(PWC_Loss, self).__init__()
        self.l_type = norm
        self.output_level = output_level
        self.num_levels = 7
        self.weights = [0.32,0.08,0.02,0.01,0.005]

        if self.l_type == 'L1': self.loss = L1()
        else: self.loss = L2()

        self.multiScales = [nn.AvgPool2d(2**l, 2**l) for l in range(self.num_levels)][::-1][:self.output_level]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, outputs, target, valid, max_flow=400):
        mag = torch.sum(target**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < max_flow)
        #args = self.args
         # if flow is normalized, every output is multiplied by its size
         # correspondingly, groundtruth should be scaled at each level
        targets = [avg_pool(target) / 2 ** (self.num_levels - l - 1) for l, avg_pool in enumerate(self.multiScales)] + [target]
        loss, epe = 0, 0
        #loss_levels, epe_levels = 
        for w, o, t in zip(self.weights, outputs, targets):
            # print(f'flow值域: ({o.min()}, {o.max()})')')
            # print(f'gt值域: ({t.min()}, {t.max()})')')
            # print(f'EPE:', EPE(o, t))')
            #print(o.shape, t.shape, "*********************")
            loss += w * self.loss(o, t)
            #epe += EPE(o, t)
            epe = torch.sum((outputs[-1] - target)**2, dim=1).sqrt()
            epe = epe.view(-1)[valid.view(-1)]
            
        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }
        
        return loss, metrics

class EpeLoss(nn.Module):
    def __init__(self, eps = 0):
        super(EpeLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, label):
        loss = ((pred - label).pow(2).sum(1) + self.eps).sqrt()
        return loss.view(loss.shape[0], -1).mean(1)


class EpeLossWithMask(nn.Module):
    def __init__(self, eps=1e-8, q=None):
        super(EpeLossWithMask, self).__init__()
        self.eps = eps
        self.q = q

    def forward(self, pred, label, mask):
        if self.q is not None:
            loss = ((pred - label).abs().sum(1) + self.eps) ** self.q
        else:
            loss = ((pred - label).pow(2).sum(1) + self.eps).sqrt()
        loss = loss * mask.squeeze(1)
        loss = loss.view(loss.shape[0], -1).sum(1) / mask.view(mask.shape[0], -1).sum(1)
        return loss


class MultiscaleEpe(nn.Module):
    def __init__(self, match='upsampling', eps = 1e-8, q = None):
        super(MultiscaleEpe, self).__init__()

        self.scales = [64, 32, 16, 8, 4]
        self.weights = [.005, .01, .02, .08, .32]
        self.match = match
        self.eps = eps
        self.q = q

    def forward(self, flow, mask, predictions):
        losses = 0
        if self.match == 'upsampling':
            for p, w, s in zip(predictions, self.weights, self.scales):
                losses += EpeLossWithMask(eps=self.eps, q=self.q)(Upsample(p, s), flow, mask) * w
        elif self.match == 'downsampling':
            for p, w, s in zip(predictions, self.weights, self.scales):
                losses += EpeLossWithMask(eps=self.eps, q=self.q)(p, Downsample(flow, s), Downsample(mask, s)) * w
        else:
            raise NotImplementedError
        return losses

class loss_function(nn.Module):
    def __init__(self, mode="loglike", sigma=25, device='cuda'):
        super(loss_function, self).__init__()
        self.mode = mode
        self.sigma = sigma
        self.device = device

    def forward(self, output, truth):
        if self.mode == "mse":
            loss = F.mse_loss(output, truth, reduction="sum") / (truth.size(0) * 2)
        elif self.mode == 'l1':
            loss = torch.abs(output - truth).mean()
        elif self.mode == "loglike":
            eps = 1e-5
            N,C,H,W = truth.shape
            mean = output[0:N, 0:C, 0:H, 0:W].permute(0,2,3,1).reshape(N, H, W, C, 1)
            var = output[0:N, C:C+int(C*(C+1)/2), 0:H, 0:W].permute(0,2,3,1).cuda()
            truth = truth.permute(0,2,3,1).reshape(N, H, W, C, 1)
            ax = torch.zeros(N, H, W, int(C*C)).to(self.device)
            I = torch.eye(C).reshape(1,1,1,C,C).repeat(N, H, W, 1, 1).to(self.device)
            idx1 = 0
            for i in range(C):
                idx2 = idx1 + C-i
                ax[0:N, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:H, 0:W, idx1:idx2]
                idx1 = idx2
            ax = ax.reshape(N, H, W, C, C)
            sigma2I = (((self.sigma**2)+eps)*I.permute(1,2,3,4,0)).permute(4,0,1,2,3)
            variance = torch.matmul(ax.transpose(3,4), ax) + sigma2I #(sigma**2)*I
            likelihood = 0.5*torch.matmul(torch.matmul((truth.to(self.device)-mean.to(self.device)).transpose(3,4).to(self.device), torch.inverse(variance.to(self.device)).to(self.device)).to(self.device), (truth.to(self.device)-mean.to(self.device)).to(self.device))
            likelihood = likelihood.reshape(N,H,W)
            likelihood += 0.5*torch.log(torch.det(variance))
            #loss = torch.mean(likelihood)
            loss = torch.mean(likelihood.mean(dim=(1,2)) - 0.1*self.sigma) 
        return loss

class Seq_L1(nn.Module):
    def __init__(self, n_frames=5, cpf=3):
        super(Seq_L1, self).__init__()
        self.n_frames = n_frames
        self.cpf = cpf
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1, reduce=False,size_average=False)

    def forward(self, input, output, target):
        loss_all = 0.0
        for i in range(self.n_frames):
            if i!= 2:
                loss = torch.abs(output[i] - target).mean()
            #loss = self.triplet_loss(input[:, (i*self.cpf):(i+1)*self.cpf, :, :], output[i], target)[0].mean()

                loss_all += loss
        return loss_all / (self.n_frames-1)
class Seq_L1_fea(nn.Module):
    def __init__(self, n_frames=5, cpf=3):
        super(Seq_L1_fea, self).__init__()
        self.n_frames = n_frames
        self.cpf = cpf
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1, reduce=False,size_average=False)

    def forward(self, fea_list):
        targets = fea_list[self.n_frames // 2]
        loss_all = 0.0
        for i in range(self.n_frames):
            if i!=(self.n_frames // 2):
                for j in range(len(fea_list[i])):
                    loss = torch.abs(fea_list[i][j] - targets[j]).mean()
            #loss = self.triplet_loss(input[:, (i*self.cpf):(i+1)*self.cpf, :, :], output[i], target)[0].mean()

                    loss_all += loss
        return loss_all / ( (self.n_frames-1) * len(targets) )

class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.
    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(
                        x_features[k] - gt_features[k],
                        p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(
                        x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) -
                        self._gram_mat(gt_features[k]),
                        p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(
                        self._gram_mat(x_features[k]),
                        self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
