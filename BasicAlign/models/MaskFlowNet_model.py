import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
import models.networks as networks
#import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, FSLoss, GradientLoss, Sequence_Loss, MultiScaleEpe, EpeLossWithMask

import torch.optim as optim
import evaluate
import os

logger = logging.getLogger('base')
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss 
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


class MaskFlowNetModel(BaseModel):
    def __init__(self, opt):
        super(MaskFlowNetModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.opt = opt
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load_raft()

        if self.is_train:
            self.netG.train()

            if opt['stage'] != 'chairs':
                self.netG.module.freeze_bn() 

            # loss
            loss_type = train_opt['pixel_criterion']
            self.loss_type = loss_type
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'sequence_loss':
                self.cri_pix = Sequence_Loss(gamma=train_opt['gamma']).to(self.device)
            elif loss_type == 'MultiScaleEpe':
                self.cri_pix = MultiScaleEpe().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            #wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

            self.optimizer_G, self.scheduler = self.fetch_optimizer(opt, self.netG)
            self.optimizers.append(self.optimizer_G)
            self.scaler = GradScaler(enabled=opt['network_G']['mixed_precision'])

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        #self.var_L = data['LQ'].to(self.device)  # LQ
        #if need_GT:
        #    self.real_H = data['GT'].to(self.device)  # GT
        self.img1, self.img2, self.flow, self.valid = [x.cuda() for x in data]

    def fetch_optimizer(self, opt, model):
        """ Create the optimizer and learning rate scheduler """
        optimizer = optim.AdamW(model.parameters(), lr=opt['train']['lr_G'], weight_decay=opt['train']['weight_decay'])
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, opt['train']['lr_G'], opt['train']['niter']+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
        
        return optimizer, scheduler
        
    def add_noise(self, image1, image2):
        stdv = np.random.uniform(0.0, 5.0)
        image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
        image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

        return image1, image2
    
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True): 
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda''' 
        batch_size = x.size()[0] 
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1 
        index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size) 
        mixed_x = lam * x + (1 - lam) * x[index,:] 
        mixed_y = lam * y + (1 - lam) * y[index,:] 
        return mixed_x, mixed_y
    
    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        
        '''add mixup operation'''
        #self.var_L, self.real_H = self.mixup_data(self.var_L, self.real_H)
        if self.opt['datasets']['train']['add_noise']:
            self.img1, self.img2 = self.add_noise(self.img1, self.img2)
            
        self.pred_flow, self.mask, _ = self.netG(self.img1, self.img2)
        if self.loss_type == 'sequence_loss':
            flow_loss, metrics = self.cri_pix(self.pred_flow, self.flow, self.valid)
        elif self.loss_type == 'MultiScaleEpe':
            flow_loss = self.cri_pix(self.flow, self.mask, self.pred_flow)
            #loss = self.l_pix_w * flow_loss
        #l_pix.backward()
        #self.optimizer_G.step()
        self.scaler.scale(flow_loss).backward()
        self.scaler.unscale_(self.optimizer_G)
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt['train']['clip'])  # 梯度裁剪

        self.scaler.step(self.optimizer_G)
        self.scheduler.step()
        self.scaler.update()

        # set log
        self.log_dict['flow_loss'] = flow_loss.item()
        if self.loss_type == 'sequence_loss':
            self.log_dict['1px'] = metrics['1px']
            self.log_dict['3px'] = metrics['3px']
            self.log_dict['5px'] = metrics['5px']
            self.log_dict['epe'] = metrics['epe']

    def test(self, opt):
        #self.netG.eval()
        
        with torch.no_grad():
            result=[]
            for val_dataset in opt['name']:
                if val_dataset == 'chairs':
                    val_dataroot = opt['dataroot_chairs']
                    val_res = evaluate.validate_chairs(self.netG.module, val_dataroot)
                    result.append(val_res)
                elif val_dataset == 'sintel':
                    val_dataroot = opt['dataroot_sintel']
                    val_res = evaluate.validate_sintel(self.netG.module, val_dataroot)
                    result.append(val_res)
                elif val_dataset == 'kitti':
                    val_dataroot = opt['dataroot_kitti']
                    val_res = evaluate.validate_kitti(self.netG.module, val_dataroot)
                    result.append(val_res)

        self.netG.train()
        if opt['stage'] != 'chairs':
            self.netG.module.freeze_bn()

        #return val_res
        return result

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
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
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

    def load_raft(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            self.netG.load_state_dict(torch.load(load_path_G))

    def save_raft(self, iter_label, network_label='G'):
        save_filename = str(iter_label) + '_' + network_label + '.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        torch.save(self.netG.state_dict(), save_path)
