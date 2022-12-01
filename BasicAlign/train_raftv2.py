from __future__ import print_function, division
import sys
#sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from models.archs.raft import RAFT
import evaluate
import data.datasets as datasets

from torch.utils.tensorboard import SummaryWriter

####add module
from models import create_model
from models.loss import Sequence_Loss
import options.options as option
from utils import util
import logging

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

def fetch_optimizer(opt, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=opt['train']['lr_G'], weight_decay=opt['train']['weight_decay'], eps=opt['train']['eta_min'])

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, opt['train']['lr_G'], opt['train']['niter']+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

SUM_FREQ = 100
VAL_FREQ = 200

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

        #### mkdir and loggers
        if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
            if resume_state is None:
                util.mkdir_and_rename(
                    opt['path']['experiments_root'])  # rename experiment folder if exists
                util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                             and 'pretrain_model' not in key and 'resume' not in key))

            # config loggers. Before it, the log will not work
            util.setup_logger('base', opt['path']['log'], opt['name'], level=logging.INFO,
                              screen=True, tofile=True)
            logger = logging.getLogger('base')
            logger.info(option.dict2str(opt))
            # tensorboard logger
            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                version = float(torch.__version__[0:3])
                if version >= 1.1:  # PyTorch 1.1
                    from torch.utils.tensorboard import SummaryWriter
                else:
                    logger.info(
                        'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                    from tensorboardX import SummaryWriter
                tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
        else:
            util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
            logger = logging.getLogger('base')

        # convert to NoneDict, which returns None for missing keys
        opt = option.dict_to_nonedict(opt)

        #### random seed
        seed = opt['train']['manual_seed']
        if seed is None:
            seed = random.randint(1, 10000)
        if rank <= 0:
            logger.info('Random seed: {}'.format(seed))
        util.set_random_seed(seed)

        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

        model_select = create_model(opt)
        model = model_select.netG
        print("Parameter Count: %d" % count_parameters(model))

        #if args.restore_ckpt is not None:
        #    model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

        # model.cuda()
        # model.train()

        if opt['stage'] != 'chairs':
            model.module.freeze_bn()

        train_loader = datasets.fetch_dataloader(opt)
        optimizer, scheduler = fetch_optimizer(opt, model)

        total_steps = 0
        scaler = GradScaler(enabled=opt['network_G']['mixed_precision'])
        logger = Logger(model, scheduler)

        # add_noise = True

        should_keep_training = True
        while should_keep_training:

            for i_batch, data_blob in enumerate(train_loader):
                optimizer.zero_grad()
                image1, image2, flow, valid = [x.cuda() for x in data_blob]

                if opt['datasets']['train']['add_noise']:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                    image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

                flow_predictions = model(image1, image2, opt['network_G']['iters'])

                # loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
                loss_me = Sequence_Loss()
                #loss, metrics = loss_me(flow_predictions, flow, valid, args.gamma)
                loss, metrics = loss_me(flow_predictions, flow, valid, 0.8)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt['train']['clip'])  # 梯度裁剪

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                logger.push(metrics)

                if total_steps % VAL_FREQ == VAL_FREQ - 1:
                    PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, opt['name'])
                    torch.save(model.state_dict(), PATH)

                    results = {}
                    for val_dataset in opt['datasets']['val']['name']:
                        if val_dataset == 'chairs':
                            results.update(evaluate.validate_chairs(model.module))
                        elif val_dataset == 'sintel':
                            results.update(evaluate.validate_sintel(model.module))
                        elif val_dataset == 'kitti':
                            results.update(evaluate.validate_kitti(model.module))

                    logger.write_dict(results)

                    model.train()
                    if opt['stage'] != 'chairs':
                        model.module.freeze_bn()

                total_steps += 1

                if total_steps > opt['train']['niter']:
                    should_keep_training = False
                    break

        logger.close()
        PATH = 'checkpoints/%s.pth' % opt['name']
        torch.save(model.state_dict(), PATH)

        return PATH

if __name__ == '__main__':
    main()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name', default='raft', help="name your experiment")
    # parser.add_argument('--stage', help="determines which dataset to use for training")
    # parser.add_argument('--restore_ckpt', help="restore checkpoint")
    # parser.add_argument('--small', action='store_true', help='use small model')
    # parser.add_argument('--validation', type=str, nargs='+')
    #
    # parser.add_argument('--lr', type=float, default=0.00002)
    # parser.add_argument('--num_steps', type=int, default=100000)
    # parser.add_argument('--batch_size', type=int, default=6)
    # parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    # parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    #
    # parser.add_argument('--iters', type=int, default=12)
    # parser.add_argument('--wdecay', type=float, default=.00005)
    # parser.add_argument('--epsilon', type=float, default=1e-8)
    # parser.add_argument('--clip', type=float, default=1.0)
    # parser.add_argument('--dropout', type=float, default=0.0)
    # parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    # parser.add_argument('--add_noise', action='store_true')
    # args = parser.parse_args()
    #
    # torch.manual_seed(1234)
    # np.random.seed(1234)
    # gpu_list = ','.join(str(x) for x in args.gpus)
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    #
    # if not os.path.isdir('checkpoints'):
    #     os.mkdir('checkpoints')

    #train(args)
