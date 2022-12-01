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
import evaluate
import data.datasets as datasets

from torch.utils.tensorboard import SummaryWriter

####add module
from models import create_model
from models.loss import Sequence_Loss
import options.options as option
from utils import util
import logging
from data import create_dataset, create_dataloader
import math
from collections import OrderedDict
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

SUM_FREQ = 100
VAL_FREQ = 200

class Logger:
    def __init__(self, model):
        self.model = model
        #self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, ".format(self.total_steps+1)
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

    #for phase, dataset_opt in opt['datasets'].items():
        #if phase == 'train':
    train_set = create_dataset(opt['datasets']['train'])
    train_size = int(math.ceil(len(train_set) / opt['datasets']['train']['batch_size']))
    total_iters = int(opt['train']['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))
    if opt['dist']:
        train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
        total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
    else:
        train_sampler = None
    train_loader = create_dataloader(train_set, opt['datasets']['train'], opt, train_sampler)
    if rank <= 0:
        logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
        logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
            #elif phase == 'val':
            #    val_set = create_dataset(dataset_opt)
            #    val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            #if rank <= 0:
            #    logger.info('Number of val images in [{:s}]: {:d}'.format(
            #        dataset_opt['name'], len(val_set)))
            #else:
            #    raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
            
    assert train_loader is not NotImplemented

    model = create_model(opt)

    #if opt['stage'] != 'chairs':
    #    model.module.freeze_bn()

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #logger_raft = Logger(model.netG)
    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    running_loss = OrderedDict()
    min_epe_loss=float('Inf')
    min_epe_step=0
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            ####training
            model.feed_data(train_data)
            model.optimize_parameters()

            #running_loss = OrderedDict()
            logs = model.get_current_log()
            #print(logs)
            for key, value in logs.items():
                if key not in running_loss:
                    running_loss[key] = 0.0
                running_loss[key] += value

            ####update learning rate
            #model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                #logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e}'.format(v)
                message += ')] '
                for k, v in running_loss.items():
                    #print(k, v, "***********************************")
                    message += '{:s}: {:.4e} '.format(k, v / opt['logger']['print_freq'])
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
                    
                running_loss = OrderedDict()

            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                if rank<=0:
                    results = model.test(opt['datasets']['val'])
                    val_message = ''
                    #print(val_result, "------------------")
                    for logs in results:
                        for k, v in logs.items():
                            val_message += '{:s}: {:.4e} '.format(k, v) 
                        val_message += '\n'
                    
                    #if epe_loss < min_epe_loss:
                    #    min_epe_loss = epe_loss
                    #    min_epe_step = current_step

                    # log
                    logger.info('# Validation # EPE result: {:s}'.format(val_message))
                    #logger.info('# Validation best EPE is : {:.4e}, best step is: {:8,d}'.format(min_epe_loss, min_epe_step))
                    # tensorboard logger
                    #if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    #    tb_logger.add_scalar('epe', epe_loss, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save_raft(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save_raft('latest')
        logger.info('End of training.')
        tb_logger.close()

if __name__ == '__main__':
    main()
