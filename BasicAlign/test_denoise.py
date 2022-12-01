import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader_davis
from models import create_model
import data.noise_davis as noise_davis
import utils.metrics as metrics

#### options
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

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    #print(phase, dataset_opt, "0000000000000000")
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader_davis(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
#test_opt = opt['datasets']['test']
for test_loader in test_loaders:
    #print(test_loader, "111111111111111111111")
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    #print(dataset_dir, "------------------")
    #util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_opt = test_loader.dataset.opt

    for data in test_loader:
        noise = noise_davis.get_noise(data['data'], dist=test_opt['noise_dist'], mode=test_opt['noise_mode'], noise_std=test_opt['noise_std'])
        model.feed_data(data['data'], noise)
        img_path = data['img_path'][0]
        img_dir = img_path.split('/')[-2]
        #print(img_dir, "*****************")
        #print(img_path, '*****************************************')
        img_name = osp.splitext(osp.basename(img_path))[0]
        #print(img_name, "****************8")

        model.test()
        visuals = model.get_current_visuals()

        output_model = visuals['output']  # uint8

        #print((data['data']+noise).shape, "111111111111") #1 15 H W 5帧输入
        #img_in = util.tensor2img((data['data']+noise)[:, (mid*cpf):((mid+1)*cpf), :, :])
        img_out = util.tensor2img(output_model)
        sample = data['data']

        mid = opt['n_frames'] // 2
        cpf = opt['network_G']['channels_per_frame']
        img_in = util.tensor2img((data['data']+noise)[:, (mid*cpf):((mid+1)*cpf), :, :])
        img_ori = util.tensor2img((data['data'])[:, (mid*cpf):((mid+1)*cpf), :, :])
        psnr = metrics.psnr(sample[:, (mid*cpf):((mid+1)*cpf), :, :], output_model)
        ssim = metrics.ssim(sample[:, (mid*cpf):((mid+1)*cpf), :, :], output_model)

        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_path = osp.join(dataset_dir, img_dir)
            util.mkdir(save_path)
            save_img_out_path = osp.join(save_path, img_name + '_out.jpg')
            save_img_in_path = osp.join(save_path, img_name + '_in.jpg')
            save_img_ori_path = osp.join(save_path, img_name + '_ori.jpg')
        if opt['save_img']:
            util.save_img(img_in, save_img_in_path)
            util.save_img(img_out, save_img_out_path)
            util.save_img(img_ori, save_img_ori_path)

        #logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
        #            format(img_name, psnr, ssim, psnr_y, ssim_y))
        logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}. '.
                    format(img_name, psnr, ssim))

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info(
        '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
            test_set_name, ave_psnr, ave_ssim))
    if test_results['psnr_y'] and test_results['ssim_y']:
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        logger.info(
            '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.
            format(ave_psnr_y, ave_ssim_y))
