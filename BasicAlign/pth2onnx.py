import sys
#sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

#from models.archs.raft import RAFT
from data.utils import flow_viz
from data.utils.utils import InputPadder

import torch.nn.functional as F
import options.options as option
import utils.util as util
from models import create_model
import logging

import time

DEVICE = 'cuda'

def load_image(imfile):
    #if img_mode == 'ppm':
    img = Image.open(imfile)
    #if img.size[0]>1280 or img.size[1]>1280:
    #    img = img.resize((1280, 1280), Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)

    img = torch.from_numpy(img).permute(2, 0, 1).float()

    return img[None].to(DEVICE)

def viz(img, flo, img_name, model_name, save_path):
    #print(img.shape, flo.shape, img.type(), flo.type())  [1 3 440 1024] [1 2 440 1024]
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()  #440 1024 2
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    #img_flo = np.concatenate([img, flo], axis=0)

    cv2.imwrite(save_path + '/' + 'target_img.png', img[:, :, [2, 1, 0]])
    cv2.imwrite(save_path + '/' + img_name + '_' + model_name + '_flo.png', flo[:, :, [2, 1, 0]])


def refine_img_torch(img, flo, img_name, dims, align_corners=True):
    flo[:, 0:1, :, :] = flo[:, 0:1, :, :] / (flo.shape[-2:][1] -1)
    flo[:, 1:2, :, :] = flo[:, 1:2, :, :] / (flo.shape[-2:][0] -1)
    flo = flo.permute(0, 2, 3, 1)
    #print(flo.max())
    H, W = img.shape[-2:]
    #flo = flo
    gridY = torch.linspace(-1, 1, steps=H).view(1, -1, 1, 1).expand(1, H, W, 1)
    gridX = torch.linspace(-1, 1, steps=W).view(1, 1, -1, 1).expand(1, H, W, 1)
    grid = torch.cat([gridX, gridY], dim=3).cuda() #[1 440 1024 2]
    #print(flo.shape, grid.shape)
    flo_up = flo + grid
    img_ref = F.grid_sample(img, flo_up, align_corners=align_corners)
    unpadder = InputPadder(dims)
    img_ref = unpadder.unpad(img_ref)

    img_ref = img_ref[0].permute(1, 2, 0).cpu().numpy()
    cv2.imwrite('result/' + img_name + '_refine_bytorch' + '.png', img_ref[:, :, [2, 1, 0]])

def refine_img_cv2(img, flow, img_name, model_name, save_path, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flow = flow[0].permute(1, 2, 0).cpu().numpy()
    #print(flow.shape)
    h_scale, w_scale = img.shape[:2]
    X, Y = np.meshgrid(np.linspace(0, w_scale-1, w_scale), np.linspace(0, h_scale-1, h_scale))
    #print(X)
    map_x = (X + flow[:, :, 0:1].squeeze()).astype(np.float32)
    map_y = (Y + flow[:, :, 1:2].squeeze()).astype(np.float32)

    ref_img = cv2.remap(img, map_x, map_y, interpolation=interpolation, borderMode=border_mode)
    cv2.imwrite(save_path + '/' + img_name + '_refine_' + model_name + '.png', ref_img[:, :, [2, 1, 0]])

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
mode_scale = opt['scale']
model_name = opt['name'] + '_' + os.path.basename(opt['path']['pretrain_model_G'].split('.')[0])

def demo():
    model = create_model(opt)
    model = model.netG.module.cuda()
    #model.to(DEVICE)
    #model.eval()
    iters_num=opt['network_G']['iters']
    dummy_input1 = torch.randn(1, 3, 1280, 1280).cuda()
    dummy_input2 = torch.randn(1, 3, 1280, 1280).cuda()
    input_names = [ "input"]
    output_names = [ "output" ]
    torch.onnx.export(model, (dummy_input1, dummy_input2), model_name+ ".onnx", 
            opset_version=12, verbose=False, input_names=input_names, output_names=output_names, keep_initializers_as_inputs=True)

if __name__ == '__main__':
    demo()
