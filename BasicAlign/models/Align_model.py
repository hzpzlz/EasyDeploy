import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
import models.networks as networks
#import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, FSLoss, GradientLoss, Sequence_Loss, Sequence_Warp, PWC_Loss, Flow_Loss
import torch.optim as optim
import evaluate
import os
import torch.nn.functional as F
from models.archs.pwcnet.modules import WarpingLayer
def refine_img_torch(img, flo, align_corners=True):
    flo[:, 0:1, :, :] = flo[:, 0:1, :, :] / (flo.shape[-2:][1] -1)
    flo[:, 1:2, :, :] = flo[:, 1:2, :, :] / (flo.shape[-2:][0] -1)
    flo = flo.permute(0, 2, 3, 1)
    H, W = img.shape[-2:]
    gridY = torch.linspace(-1, 1, steps=H).view(1, -1, 1, 1).expand(1, H, W, 1)
    gridX = torch.linspace(-1, 1, steps=W).view(1, 1, -1, 1).expand(1, H, W, 1)
    grid = torch.cat([gridX, gridY], dim=3).cuda() #[1 440 1024 2]
    flo_up = flo + grid
    img_ref = F.grid_sample(img, flo_up, align_corners=align_corners)
    return img_ref

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


class AlignModel(BaseModel):
    def __init__(self, opt):
        super(AlignModel, self).__init__(opt)

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
        #self.load_raft()
        self.load()

        if self.is_train:
            self.netG.train()

            if (opt['stage'] != 'chairs') and (opt['network_G']['which_model_G'] in ['RAFT']):
                self.netG.module.freeze_bn() 

            # loss
            if train_opt['pixel_criterion'] is not None:
                pixel_loss_type = train_opt['pixel_criterion']
                self.pixel_loss_type = pixel_loss_type
                if pixel_loss_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif pixel_loss_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif pixel_loss_type == 'cb':
                    self.cri_pix = CharbonnierLoss().to(self.device)
                    
                self.l_pix_w = train_opt['pixel_weight']
            else:
                self.pixel_loss_type = None

            if train_opt['flow_criterion'] is not None:
                flow_loss_type = train_opt['flow_criterion']
                self.flow_loss_type = flow_loss_type
                #print(self.flow_loss_type, "**************************************")
                if flow_loss_type == 'sequence_loss':
                    self.cri_flow = Sequence_Loss(gamma=train_opt['gamma']).to(self.device)
                elif flow_loss_type == 'pwc_loss':
                    self.cri_flow = PWC_Loss(output_level=opt['network_G']['output_level']).to(self.device)
                elif flow_loss_type == 'flow_loss':
                    self.cri_flow = Flow_Loss().to(self.device)
                    
                self.l_flow_w = train_opt['flow_weight']
            else:
                self.flow_loss_type = None

            if train_opt['warp_criterion'] is not None:
                warp_loss_type = train_opt['warp_criterion']
                self.warp_loss_type = warp_loss_type
                #print(self.flow_loss_type, "**************************************")
                if warp_loss_type == 'sequence_warp':
                    self.cri_warp = Sequence_Warp(gamma=train_opt['gamma']).to(self.device)
                    
                self.l_warp_w = train_opt['warp_weight']
            else:
                self.warp_loss_type = None
            #else:
            #    raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            #self.l_pix_w = train_opt['pixel_weight']

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
        #optimizer = optim.AdamW(model.parameters(), lr=opt['train']['lr_G'], weight_decay=opt['train']['weight_decay'])
        optimizer = optim.Adam(model.parameters(), lr=opt['train']['lr_G'], weight_decay=opt['train']['weight_decay'], betas=(opt['train']['beta1'], opt['train']['beta2']))

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, opt['train']['lr_G'], opt['train']['niter']+100, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
        
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
        self.feature_loss = None
        self.optimizer_G.zero_grad()
        
        '''add mixup operation'''
        #self.var_L, self.real_H = self.mixup_data(self.var_L, self.real_H)
        if self.opt['datasets']['train']['add_noise']:
            self.img1, self.img2 = self.add_noise(self.img1, self.img2)
            
        if self.opt['network_G']['which_model_G'] in ['RMOF', 'RMOF_v1']:
            self.pred_flow, self.feature_loss = self.netG(self.img1, self.img2)
        else:
            self.pred_flow = self.netG(self.img1, self.img2)
        loss_total = 0
        if self.flow_loss_type in ['sequence_loss', 'pwc_loss', 'flow_loss']:
            flow_loss, metrics = self.cri_flow(self.pred_flow, self.flow, self.valid)
            flow_loss = self.l_flow_w * flow_loss
            loss_total += flow_loss
        if self.pixel_loss_type is not None:
            H, W = self.img1.shape[-2:]
            if self.opt['network_G']['which_model_G'] in ['pwcplus']:
                flo = self.pred_flow#.permute(0, 2, 3, 1)
            else:
                flo = 2*self.pred_flow[-1]#.permute(0, 2, 3, 1)
            warp_img = refine_img_torch(self.img2, flo)

            pix_loss = self.l_pix_w * self.cri_pix(self.img1, warp_img)
            loss_total += pix_loss
        if self.warp_loss_type is not None:
            warp_loss = self.l_warp_w * self.cri_warp(self.warp_imgs, self.img1)
            loss_total += warp_loss
        if self.feature_loss is not None:
            loss_total += self.feature_loss
            
        #l_pix.backward()
        #self.optimizer_G.step()
        self.scaler.scale(loss_total).backward()
        self.scaler.unscale_(self.optimizer_G)
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt['train']['clip'])  # 梯度裁剪

        self.scaler.step(self.optimizer_G)
        self.scheduler.step()
        self.scaler.update()

        # set log
        self.log_dict['loss_total'] = loss_total.item()
        if self.pixel_loss_type is not None:
            self.log_dict['pix_loss'] = pix_loss.item()
        if self.flow_loss_type is not None:
            self.log_dict['flow_loss'] = flow_loss.item()
            self.log_dict['1px'] = metrics['1px']
            self.log_dict['3px'] = metrics['3px']
            self.log_dict['5px'] = metrics['5px']
            self.log_dict['epe'] = metrics['epe']
        if self.warp_loss_type is not None:
            self.log_dict['warp_loss'] = warp_loss.item()
        if self.feature_loss is not None:
            self.log_dict['feature_loss'] = self.feature_loss.item()

    def test(self, opt):
        #self.netG.eval()
        opt_val = opt['datasets']['val']
        with torch.no_grad():
            result=[]
            for val_dataset in opt_val['name']:
                if val_dataset == 'chairs':
                    val_dataroot = opt_val['dataroot_chairs']
                    val_res = evaluate.validate_chairs(self.netG.module, val_dataroot, self.opt['network_G']['which_model_G'], self.opt['scale'])
                    result.append(val_res)
                elif val_dataset == 'sintel':
                    val_dataroot = opt_val['dataroot_sintel']
                    val_res = evaluate.validate_sintel(self.netG.module, val_dataroot)
                    result.append(val_res)
                elif val_dataset == 'kitti':
                    val_dataroot = opt_val['dataroot_kitti']
                    val_res = evaluate.validate_kitti(self.netG.module, val_dataroot)
                    result.append(val_res)

        self.netG.train()
        if (opt['stage'] != 'chairs') and (opt['network_G']['which_model_G'] in ['RAFT']):
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
