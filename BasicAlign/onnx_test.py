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
from torch.nn.parallel import DistributedDataParallel
import models.Align_model as am

import models.archs.raft as raft
import onnxruntime
from models.archs.pwcnet.modules import warp

DEVICE = 'cuda'

def load_image(imfile):
    #if img_mode == 'ppm':
    img = Image.open(imfile)
    #if img.size[0]>1280 or img.size[1]>1280:
    #    img = img.resize((1280, 1280), Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)

    img = torch.from_numpy(img).permute(2, 0, 1).float()

    return img[None].to(DEVICE)

def vis(fea, img_name, model_name, save_path, is_norm):
    mask = fea[0].permute(1,2,0).cpu().numpy()*255.  #440 1024 2
    
    cv2.imwrite(save_path + '/' + img_name + '_' + model_name + '_mask.png', mask)

def viz_flow(flo, img_name, model_name, save_path, is_norm):
    flo = flo[0].transpose(1,2,0)  #440 1024 2
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite(save_path + '/' + img_name + '_' + model_name + '_flo_onnx.png', flo[:, :, [2, 1, 0]])

def viz(img, warp_img, flo, img_name, model_name, save_path, is_norm):
    #print(img.shape, flo.shape, img.type(), flo.type())  [1 3 440 1024] [1 2 440 1024]
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()  #440 1024 2
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    #img_flo = np.concatenate([img, flo], axis=0)

    if is_norm:
        cv2.imwrite(save_path + '/' + img_name + '_target_img.png', img[:, :, [2, 1, 0]]*255.)
        diffs = cv2.absdiff(warp_img, img[:, :, [2, 1, 0]]*255.) * 5
    else:
        cv2.imwrite(save_path + '/' + img_name + '_target_img.png', img[:, :, [2, 1, 0]])
        diffs = cv2.absdiff(warp_img, img[:, :, [2, 1, 0]]) * 5

    diffs = cv2.cvtColor(diffs, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(save_path + '/' + img_name + '_' + model_name + '_flo.png', flo[:, :, [2, 1, 0]])
    cv2.imwrite(save_path + '/' + img_name + '_' + model_name + '_diff.png', diffs)


def refine_img_torch(img, flo, img_name, model_name, save_path, is_norm, align_corners=True):
    H, W = img.shape[-2:]

    #1
    # xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    # yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    # xx = xx.view(1, 1, H, W).repeat(1, 1, 1, 1)
    # yy = yy.view(1, 1, H, W).repeat(1, 1, 1, 1)
    # grid = torch.cat((xx, yy), 1).float().cuda()
    # #print(grid)
    # #print(grid)
    # vgrid = grid + flo
    # vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :].clone() / (W-1) - 1.0
    # vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :].clone() / (H-1) - 1.0
    # vgrid = vgrid.permute(0, 2, 3, 1)

    #2
    # flo[:, 0:1, :, :] = 2*flo[:, 0:1, :, :] / (flo.shape[-2:][1] - 1)
    # flo[:, 1:2, :, :] = 2*flo[:, 1:2, :, :] / (flo.shape[-2:][0] - 1)
    # flo = flo.permute(0, 2, 3, 1)
    # gridY = torch.linspace(-1, 1, steps=H).view(1, -1, 1, 1).expand(1, H, W, 1)
    # gridX = torch.linspace(-1, 1, steps=W).view(1, 1, -1, 1).expand(1, H, W, 1)
    # grid = torch.cat([gridX, gridY], dim=3).cuda() #[1 440 1024 2]:q

    # vgrid = flo + grid
    # img_ref = F.grid_sample(img, vgrid, align_corners=align_corners)

    #3
    img_ref = warp(img, flo)

    ref_img = img_ref[0].permute(1, 2, 0).cpu().numpy()
    if is_norm:
        output = ref_img[:, :, [2, 1, 0]]*255.
        out_path = save_path + '/' + img_name + '_refine_' + model_name + '_norm_torch.png'
    else:
        output = ref_img[:, :, [2, 1, 0]]
        out_path = save_path + '/' + img_name + '_refine_' + model_name + '_torch.png'

    cv2.imwrite(out_path, output)
    return output

def refine_img_cv2(img, flow, img_name, model_name, save_path, is_norm, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flow = flow[0].permute(1, 2, 0).cpu().numpy()
    #print(flow.shape)
    h_scale, w_scale = img.shape[:2]
    X, Y = np.meshgrid(np.linspace(0, w_scale-1, w_scale), np.linspace(0, h_scale-1, h_scale))
    #print(X)
    map_x = (X + flow[:, :, 0:1].squeeze()).astype(np.float32)
    map_y = (Y + flow[:, :, 1:2].squeeze()).astype(np.float32)

    ref_img = cv2.remap(img, map_x, map_y, interpolation=interpolation, borderMode=border_mode)
    if is_norm:
        output = ref_img[:, :, [2, 1, 0]]*255.
        out_path = save_path + '/' + img_name + '_refine_' + model_name + '_norm_cv2.png'
    else:
        output = ref_img[:, :, [2, 1, 0]]
        out_path = save_path + '/' + img_name + '_refine_' + model_name + '_cv2.png'
    cv2.imwrite(out_path, output)

    return output 

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
save_dir = opt['path']['results_root']

def load_network_forirrpwc(load_path, network, strict=True):
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
        print("1111111111111111111111111111")
    load_net = torch.load(load_path)
    print(load_net, "22222222222222222222222222222222222")
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net['state_dict'].items():
        print(k, "************************************")
        if k.startswith('_model.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)

def demo():
    model = create_model(opt)
    model = model.netG.eval()

    if opt['network_G']['which_model_G'] in ['RAFT', 'RAFT_STN']:
        iters_num=opt['network_G']['iters']
    if opt['network_G']['which_model_G'] in ['PWCNet']:
        output_level=opt['network_G']['output_level']
    H, W = tuple(int(i) for i in opt['datasets']['test']['img_size'])
    dummy_input1 = torch.randn(1, 3, H, W).cuda()
    dummy_input2 = torch.randn(1, 3, H, W).cuda()
    print(dummy_input1.shape, "///////////")

    output_names = ["output"]
    input_names = ["input1", "input2"] 
    onnx_model_path = save_dir + '/' + model_name+ ".onnx"
    print(onnx_model_path, "oooooooooooooooooooooooooooooooooooooooooo")
    torch.onnx.export(model.module, (dummy_input1, dummy_input2), onnx_model_path, 
            opset_version=11, verbose=False, input_names=input_names, output_names=output_names)#, use_external_data_format=True)#, keep_initializers_as_inputs=True)
    print("--------------- Convert to onnx successfully!!! -------------------------")

    #model = am.AlignModel(opt)
    #model.netG = torch.nn.DataParallel(irrpwc.PWCNet().cuda())
    #model.load()
    #model=model.netG

    ###for pwcirr
    #model = torch.nn.DataParallel(irrpwc.PWCNet())
    #load_network(model_path, model)

    with torch.no_grad():
        images = glob.glob(os.path.join(data_path, '*.png')) + \
                 glob.glob(os.path.join(data_path, '*.jpg')) + \
                 glob.glob(os.path.join(data_path, '*.ppm'))

        images = sorted(images)
        #print(images)
        num=0

        spend_time=0.0
        is_norm = False  #norm不行
        #iters_num=20
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            #print(imfile1, imfile2, '***')
            img_name = os.path.basename(imfile1).split('.')[0]
            #print(img_name)
            if is_norm: 
                image1 = load_image(imfile1)/255.
                image2 = load_image(imfile2)/255.
            else:
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

            print(mode_scale, "mmmmmmmmmmmmm")
            padder = InputPadder(image1.shape, mode_scale)
            image1, image2 = padder.pad(image1, image2)

            start_time=time.time()
            dims = image1.shape
            print(dims, "*****************************")
            #feas, flow_up = model(image1, image2, test_mode=True)  #对应的方向可以改 得到的光流图和是和image1重叠的
            flow_up = model(image1, image2, test_mode=True)  #对应的方向可以改 得到的光流图和是和image1重叠的
            #flow_up = model(image2, image1, test_mode=True)['flow']  #对应的方向可以改 得到的光流图和是和image1重叠的
            print(flow_up.permute(0, 2, 3, 1)) #[1 2 55 128] 下采样的图  flow_up是放大以后的图 flow_up: [1 2 440 1024]
            #print(feas.shape) #[1 2 55 128] 下采样的图  flow_up是放大以后的图 flow_up: [1 2 440 1024]

            ###for onnx
            sess = onnxruntime.InferenceSession(onnx_model_path)
            input1 = sess.get_inputs()[0].name
            input2 = sess.get_inputs()[1].name
            if opt['network_G']['which_model_G'] in ['RAFT', 'RAFT_STN']:
                output_name = sess.get_outputs()[iters_num-1].name
            elif opt['network_G']['which_model_G'] in ['PWCNet']:
                output_name = sess.get_outputs()[output_level].name
                print(output_name, "out name")
            elif opt['network_G']['which_model_G'] in ['pwcnethty']:
                output_name = sess.get_outputs()[0].name
                print(output_name, "out name")
            out_onnx = sess.run([output_name], {input1: image1.cpu().numpy(), input2: image2.cpu().numpy()})
            print(out_onnx[0].shape, len(out_onnx), "oooooooooooooooooooooooo")
            flow_onnx = out_onnx[0]#[0].permute(1,2,0)

            unpadder = InputPadder(dims, mode_scale)
            image1 = unpadder.unpad(image1)
            image2 = unpadder.unpad(image2)
            flow_up = unpadder.unpad(flow_up)
            flow_onnx = unpadder.unpad(flow_onnx)

            warp_img = refine_img_cv2(image2, flow_up.clone(), img_name, model_name, save_dir, is_norm)   #对应的方向可以改
           # warp_img = refine_img_torch(image2, flow_up.clone(), img_name, model_name, save_dir, is_norm)   #对应的方向可以改

            end_time = time.time()
            spend_time += end_time - start_time
            viz(image1, warp_img, flow_up, img_name, model_name, save_dir, is_norm)
            viz_flow(flow_onnx, img_name, model_name, save_dir, is_norm)
            #if opt['network_G']['which_model_G'] in ['MaskFlowNet_hzp', 'MaskFlowNet_hzp_v2']:
             #   vis(feas, img_name, model_name, save_dir, is_norm)

            num+=1
        if opt['network_G']['which_model_G'] in ['RAFT', 'RAFT_STN']:
            print("img num is:", num, "; iters_num is:", iters_num, "; average time is: ", spend_time / num)
        else:
            print("img num is:", num, "; average time is: ", spend_time / num)


if __name__ == '__main__':
    demo()
