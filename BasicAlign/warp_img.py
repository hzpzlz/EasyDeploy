import sys
#sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from collections import OrderedDict

#from models.archs.raft import RAFT
from data.utils import flow_viz
from data.utils.utils import InputPadder

import torch.nn.functional as F
import options.options as option
import utils.util as util
from models import create_model
import logging

import time
import models.archs.pwcirr.IRR_PWC as irrpwc
import models.archs.xaba.xabav1 as xabav1
import models.archs.htypwc.pwcnet as pwcnet
from torch.nn.parallel import DistributedDataParallel
import models.Align_model as am

import models.archs.raft as raft
from thop import profile

DEVICE = 'cuda'

def load_image(imfile):
    img = Image.open(imfile)
    img = np.array(img).astype(np.uint8)

    img = torch.from_numpy(img).permute(2, 0, 1).float()

    return img[None].to(DEVICE)

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

data_path = opt['datasets']['test']['data_root']
model_path = opt['path']['pretrain_model_G']
print(model_path, "-------------------")
mode_scale = opt['scale']
model_name = opt['name'] + '_' + os.path.basename(opt['path']['pretrain_model_G'].split('.')[0])
print(model_name)

raw_input=opt['network_G']['raw_input']
#mymodel = xabav1.Net()
#mymodel = pwcnet.PWCNet()
#input_tmp1 = torch.rand(1,3,800,800)
#input_tmp2 = torch.rand(1,3,800,800)
#flops, params = profile(mymodel, inputs=(input_tmp1, input_tmp2))
#print(flops, params, '*******************************')

def demo():
    model = create_model(opt)
    model = model.netG.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(data_path, '*.png')) + \
                 glob.glob(os.path.join(data_path, '*.jpg')) + \
                 glob.glob(os.path.join(data_path, '*.ppm'))

        images = sorted(images)
        #print(images)
        num=0

        spend_time=0.0
        is_norm = False  #norm不行
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            img_name = os.path.basename(imfile2).split('.')[0]
            print(img_name)
            if is_norm: 
                image1 = load_image(imfile1)/255.
                image2 = load_image(imfile2)/255.
            else:
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, mode_scale)
            image1, image2 = padder.pad(image1, image2)

            align_img = model(image2, image1)  #对应的方向可以改 得到的光流图和是和image1重叠的
            align_img = padder.unpad(align_img)

            #if raw_input==False:
            out = align_img[0].permute(1,2,0)[:, :, 0:3].cpu().numpy()
            #else:
            #    out = align_img[0].permute(1,2,0).cpu().numpy()

            save_dir = opt['path']['results_root']
            cv2.imwrite(os.path.join(save_dir, img_name + '_warp_' + model_name + '.jpg'), out[..., ::-1])   #对应的方向可以改

            num+=1

if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--model', help="restore checkpoint")
#    parser.add_argument('--path', help="dataset for evaluation")
#    parser.add_argument('--iters_num', default=20, type=int, help='cycles of gru')
#    parser.add_argument('--small', action='store_true', help='use small model')
#    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
#    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
#    args = parser.parse_args()
#
    demo()

    #ori_img('/home/hzp/codes/RAFT/frameandflow/00001_img1.ppm', '/home/hzp/codes/RAFT/frameandflow/00001_img2.ppm', '/home/hzp/codes/RAFT/frameandflow/00001_flow.flo')
