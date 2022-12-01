import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, FSLoss, GradientLoss, SSIM, PerceptualLoss

logger = logging.getLogger('base')

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, scale=8, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // scale) + 1) * scale - self.ht) % scale
        pad_wd = (((self.wd // scale) + 1) * scale - self.wd) % scale
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

class AlignDenoiseModel(BaseModel):
    def __init__(self, opt):
        super(AlignDenoiseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            self.loss_type = loss_type
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss(eps=0.1).to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            self.ssim_flag = train_opt['ssim_use']
            self.ssim_weight = train_opt['ssim_weight']
            if self.ssim_flag:
                self.ssim_loss = SSIM().to(self.device)

            self.perp_flag = train_opt['perp_use']
            perp_weight = train_opt['perp_weight']
            if self.perp_flag:
                layer_weights = {'relu1_1': 3.125e-2, 'relu2_1': 6.25e-2, 'relu3_1': 0.125, 'relu4_1': 0.25, 'relu5_1': 1.0}
                self.perp_loss = PerceptualLoss(layer_weights, use_input_norm=False, perceptual_weight=perp_weight).to(self.device)


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.img_ref = data['ref'].to(self.device)  # img_ref
        self.img_base = data['base'].to(self.device)  # img_base
        self.GT = data['GT'].to(self.device)  #GT
    
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True): 
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda''' 
        batch_size = x.size()[0] 
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1 
        index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size) 
        mixed_x = lam * x + (1 - lam) * x[index,:] 
        mixed_y = lam * y + (1 - lam) * y[index,:] 
        return mixed_x, mixed_y
    
    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        
        '''add mixup operation'''
        # self.var_L, self.real_H = self.mixup_data(self.var_L, self.real_H)
        
        self.img_align = self.netG(self.img_ref, self.img_base)
        if self.loss_type == 'fs':
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H) + self.l_fs_w * self.cri_fs(self.fake_H, self.real_H)
        elif self.loss_type == 'grad':
            l1 = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            lg = self.l_grad_w * self.gradloss(self.fake_H, self.real_H)
            l_pix = l1 + lg
        elif self.loss_type == 'grad_fs':
            l1 = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            lg = self.l_grad_w * self.gradloss(self.fake_H, self.real_H)
            lfs = self.l_fs_w * self.cri_fs(self.fake_H, self.real_H)
            l_pix = l1 + lg + lfs
        else:
            l_pix = self.l_pix_w * self.cri_pix(self.img_align, self.GT)

        if self.ssim_flag:
            l_ssim = self.ssim_loss(self.img_align, self.GT)
            l_pix += (1 - l_ssim) * self.ssim_weight
        if self.perp_flag:
            aligns = torch.cat([self.img_align[:, 0:1, :, :], (self.img_align[:, 1:2, :, : ] + self.img_align[:, 2:3, :, : ])/2, self.img_align[:, 3:4, :, : ]], 1)
            bases = torch.cat([self.GT[:, 0:1, :, :], (self.GT[:, 1:2, :, : ] + self.GT[:, 2:3, :, : ])/2, self.GT[:, 3:4, :, : ]], 1)
            l_perp = self.perp_loss(aligns, bases)
            l_pix += l_perp[0]
        l_pix.backward()
        #torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 1.0)  # 梯度裁剪

        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        if self.loss_type == 'grad':
            self.log_dict['l_1'] = l1.item()
            self.log_dict['l_grad'] = lg.item()
        if self.loss_type == 'grad_fs':
            self.log_dict['l_1'] = l1.item()
            self.log_dict['l_grad'] = lg.item()
            self.log_dict['l_fs'] = lfs.item()
        if self.ssim_flag:
            self.log_dict['ssim'] = l_ssim.item()
        if self.perp_flag:
            self.log_dict['perp'] = l_perp[0].item()


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            padder = InputPadder(self.img_base.shape, self.opt['scale'], 'raw')
            self.img_ref, self.img_base = padder.pad(self.img_ref, self.img_base)
            self.img_align = self.netG(self.img_ref, self.img_base)
            self.img_align = padder.unpad(self.img_align)
            self.img_base = padder.unpad(self.img_base)
            #self.img_ref = padder.unpad(self.img_ref)
        self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['GT'] = self.GT.detach()[0].float().cpu()
        out_dict['img_align'] = self.img_align.detach()[0].float().cpu()
        out_dict['img_base'] = self.img_base.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
    
#     def load(self):
#         load_path_G_1 = self.opt['path']['pretrain_model_G_1']
#         load_path_G_2 = self.opt['path']['pretrain_model_G_2']
#         load_path_Gs=[load_path_G_1, load_path_G_2]
        
#         load_path_G = self.opt['path']['pretrain_model_G']
#         if load_path_G is not None:
#             logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
#             self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
#         if load_path_G_1 is not None:
#             logger.info('Loading model for 3net [{:s}] ...'.format(load_path_G_1))
#             logger.info('Loading model for 3net [{:s}] ...'.format(load_path_G_2))
#             self.load_network_part(load_path_Gs, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
