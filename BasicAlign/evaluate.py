import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import data.datasets as datasets
from data.utils import flow_viz
from data.utils import frame_utils

from models.archs.raft.raft import RAFT
from data.utils.utils import InputPadder, forward_interpolate

from collections import OrderedDict
from utils import util
import cv2

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

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

@torch.no_grad()
def validate_chairs_img(model, val_dataroot, raw_input=False):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    psnr_list = []
    logs = OrderedDict()

    val_dataset = datasets.FlyingChairs(split='validation', root=val_dataroot)
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, scale=80, mode='chairs')
        image1, image2 = padder.pad(image1, image2)
        align_img = model(image2, image1)

        image1 = padder.unpad(image1)
        align_img = padder.unpad(align_img)
        if raw_input==False:
            img1_mean = torch.mean(image1, dim=1, keepdim=True)
            image1 = torch.cat([image1, img1_mean], dim=1)
        
        gt_img = image1[0].permute(1,2,0).cpu().numpy()
        al_img = align_img[0].permute(1,2,0).cpu().numpy()

        psnr = util.calculate_psnr(gt_img, al_img)
        psnr_list.append(psnr)
    avg_psnr = np.mean(psnr_list)

    logs['avg_psnr'] = avg_psnr.item()

    return logs


@torch.no_grad()
def validate_chairs(model, val_dataroot, model_name, scale):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    logs = OrderedDict()

    val_dataset = datasets.FlyingChairs(split='validation', root=val_dataroot)
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        if model_name in ['pwcplus']:
            padder = InputPadder(image1.shape, scale, 'chairs')
            image1, image2 = padder.pad(image1, image2)
            flow_pr = model(image1, image2)
            flow_pr = padder.unpad(flow_pr)
        else:
            #_, flow_pr = model(image1, image2, test_mode=True)
            flow_pr = model(image1, image2, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    #return epe
    logs['chairs-epe'] = epe.item()
    #return {'chairs-epe': epe}
    return logs

@torch.no_grad()
def validate_sintel(model, val_dataroot):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}

    logs = OrderedDict()

    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', root=val_dataroot, dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            #flow_low, flow_pr = model(image1, image2, test_mode=True)
            flow_pr = model(image1, image2, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        #print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        #results[dstype] = np.mean(epe_list)
        save_name = 'sintel_' + dstype
        logs[save_name] = np.mean(epe_list).item()

    #return results
    #return {'clean-epe': results['clean'], 'final-epe': results['final']}
    return logs

@torch.no_grad()
def validate_kitti_img(model, val_dataroot):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='testing', root=val_dataroot)

    psnr_list = []
    logs = OrderedDict()

    for val_id in range(len(val_dataset)):
        image1, image2, _ = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, scale=80, mode='kitti')
        image1, image2 = padder.pad(image1, image2)
        align_img = model(image2, image1)

        image1 = padder.unpad(image1)
        align_img = padder.unpad(align_img)

        gt_img = image1[0].permute(1,2,0).cpu().numpy()
        al_img = align_img[0].permute(1,2,0).cpu().numpy()

        psnr = util.calculate_psnr(gt_img, al_img)
        psnr_list.append(psnr)
    avg_psnr = np.mean(psnr_list)
    logs['avg_psnr'] = avg_psnr.item()

    return logs

@torch.no_grad()
def validate_kitti(model, val_dataroot):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='testing', root=val_dataroot)
    logs = OrderedDict()

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        #flow_low, flow_pr = model(image1, image2, test_mode=True)
        flow_pr = model(image1, image2, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    logs['kitti-epe'] = epe.item()
    logs['kitti-f1'] = f1.item()
    #print("Validation KITTI: %f, %f" % (epe, f1))
    #return {'kitti-epe': epe, 'kitti-f1': f1}
    return logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)


