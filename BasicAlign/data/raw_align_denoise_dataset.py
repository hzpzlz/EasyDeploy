import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import glob
import os
from .preproc.raw import pack_raw_to_4ch, rescaling, bayer_to_offsets
from .util import augment_list
from .local_motion import global_shift, local_motion, PngShow

def get_full_dirs(dir):
    dir_list = []
    batch_lists = os.listdir(dir)
    for i in batch_lists:
        scenes = os.listdir(os.path.join(dir, i))
        for j in scenes:
            dir_list.append(os.path.join(dir, i, j))
    return dir_list

class RawAlignDenoiseDataset(data.Dataset):
    def __init__(self, opt):
        super(RawAlignDenoiseDataset, self).__init__()
        self.opt = opt
        self.n_frames = opt['n_frames']
        self.all_path = get_full_dirs(opt['dataroot'])

        self.get_imglists()

    def get_imglists(self):
        self.img_lists = []
        for i, data_dir in enumerate(self.all_path):
            base_path = glob.glob(os.path.join(data_dir, "IMG_2022*req[14*.RawPlain16LSB1"))[0]
            random_start_idx = 2
            ref_path = sorted(glob.glob(os.path.join(data_dir, "*.RawPlain16LSB1")))
            ref_path = ref_path[random_start_idx:random_start_idx+self.n_frames]

            base_img = np.fromfile(base_path, dtype=np.uint16).reshape(self.opt['inheight'], self.opt['inwidth'])
            ref_img = [np.fromfile(ref_idx, dtype=np.uint16).reshape(self.opt['inheight'], self.opt['inwidth']) for ref_idx in ref_path]

            base_img = rescaling(base_img, self.opt['black_level'], self.opt['white_level'])
            base_img = np.clip(base_img, 0.0, 1.0)
            ref_img = [rescaling(x, self.opt['black_level'], self.opt['white_level']) for x in ref_img]
            ref_img = [np.clip(x, 0.0, 1.0) for x in ref_img]

            offset = bayer_to_offsets(self.opt['target_pattern'])
            img_base = pack_raw_to_4ch(base_img, offset)
            img_ref = [pack_raw_to_4ch(x, offset) for x in ref_img]

            self.img_lists.append([img_ref, img_base, base_path])

    def __getitem__(self, index):
        img_size = self.opt['image_size']
        img_ref, GT, base_path = self.img_lists[index]

        if self.opt['phase'] == 'train':
            H,W = GT.shape[0:2]
            rnd_h = random.randint(0, max(0, H - img_size[0]))
            rnd_w = random.randint(0, max(0, W - img_size[1]))

            GT = GT[rnd_h:rnd_h+img_size[0], rnd_w:rnd_w+img_size[1], :]
            img_ref = [x[rnd_h:rnd_h+img_size[0], rnd_w:rnd_w+img_size[1], :] for x in img_ref]

        img_ref = global_shift(img_ref, self.n_frames)
        img_ref = local_motion(img_ref, self.n_frames, 10)
        img_ref, GT = augment_list(img_ref + [GT], self.opt['use_flip'], self.opt['use_rot'])
        # for i, frame in enumerate(img_ref):
        #     PngShow(frame, 'rggb',os.path.join('/home/hzp/codes/BasicAlign/data/' + '_localmotion_' + str(i) + ".png"), 10)
        # PngShow(img_base, 'rggb', os.path.join('/home/hzp/codes/BasicAlign/data/' + '_base_.png'), 10)

        GT = torch.from_numpy(np.ascontiguousarray(np.transpose(GT, (2, 0, 1)))).float()
        img_ref = torch.cat([torch.from_numpy(np.ascontiguousarray(np.transpose(x, (2, 0, 1)))).float() for x in img_ref[1:]], dim=0)
        img_base = torch.from_numpy(np.ascontiguousarray(np.transpose(img_ref[0], (2, 0, 1)))).float()

        return {'base': img_base, 'ref': img_ref, 'GT': GT, 'base_path': base_path}


    # def __getitem__(self, index):
    #     img_size = self.opt['image_size']
    #     data_dir = self.all_path[index]
    #
    #     base_path = glob.glob(os.path.join(data_dir, "IMG_2022*req[14*.RawPlain16LSB1"))[0]
    #
    #     random_start_idx = 2
    #     ref_path = sorted(glob.glob(os.path.join(data_dir, "*.RawPlain16LSB1")))
    #     ref_path = ref_path[random_start_idx:random_start_idx+self.n_frames]
    #
    #     base_img = np.fromfile(base_path, dtype = np.uint16).reshape(self.opt['inheight'], self.opt['inwidth'])
    #     ref_img = [np.fromfile(ref_idx, dtype = np.uint16).reshape(self.opt['inheight'], self.opt['inwidth']) for ref_idx in ref_path]
    #
    #     base_img = rescaling(base_img, self.opt['black_level'], self.opt['white_level'])
    #     base_img = np.clip(base_img, 0.0, 1.0)
    #
    #     ref_img = [rescaling(x, self.opt['black_level'], self.opt['white_level']) for x in ref_img]
    #     ref_img = [np.clip(x, 0.0, 1.0) for x in ref_img]
    #
    #     offset = bayer_to_offsets(self.opt['target_pattern'])
    #
    #     img_base = pack_raw_to_4ch(base_img, offset)
    #     img_ref = [pack_raw_to_4ch(x, offset) for x in ref_img]
    #
    #     if self.opt['phase'] == 'train':
    #         H,W = img_base.shape[0:2]
    #         rnd_h = random.randint(0, max(0, H - img_size[0]))
    #         rnd_w = random.randint(0, max(0, W - img_size[1]))
    #
    #         img_base = img_base[rnd_h:rnd_h+img_size[0], rnd_w:rnd_w+img_size[1], :]
    #         img_ref = [x[rnd_h:rnd_h+img_size[0], rnd_w:rnd_w+img_size[1], :] for x in img_ref]
    #
    #     # for i, frame in enumerate(img_ref):
    #     #     PngShow(frame, 'rggb',os.path.join('/home/hzp/codes/BasicAlign/data/' + '_oriimg' + str(i) + ".png"), 10)
    #     img_ref = global_shift(img_ref, self.n_frames)
    #     # for i, frame in enumerate(img_ref):
    #     #     PngShow(frame, 'rggb',os.path.join('/home/hzp/codes/BasicAlign/data/' + '_globalmotion_' + str(i) + ".png"), 10)
    #     img_ref = local_motion(img_ref, self.n_frames, 10)
    #     # for i, frame in enumerate(img_ref):
    #     #     PngShow(frame, 'rggb',os.path.join('/home/hzp/codes/BasicAlign/data/' + '_localmotion_' + str(i) + ".png"), 10)
    #     #PngShow(img_base, 'rggb', os.path.join('/home/hzp/codes/BasicAlign/data/' + '_base_.png'), 10)
    #
    #     img_ref, img_base = augment_list(img_ref+[img_base], self.opt['use_flip'], self.opt['use_rot'])
    #     #for i, frame in enumerate(img_ref):
    #     #    PngShow(frame, 'rggb',os.path.join('/home/hzp/codes/BasicAlign/data/' + '_augment_' + str(i) + ".png"), 10)
    #     #PngShow(img_base, 'rggb', os.path.join('/home/hzp/codes/BasicAlign/data/' + '_base_augment_.png'), 10)
    #
    #     img_base = torch.from_numpy(np.ascontiguousarray(np.transpose(img_base, (2, 0, 1)))).float()
    #     img_ref = torch.cat([torch.from_numpy(np.ascontiguousarray(np.transpose(x, (2, 0, 1)))).float() for x in img_ref], dim=0)
    #
    #     return {'base': img_base, 'ref': img_ref, 'base_path': base_path, 'ref_path': ref_path}

    def __len__(self):
        return len(self.all_path)

if __name__ == '__main__':
    opt = {'phase': 'train',
           'dataroot': '/home/hzp/datasets/Align_Denoise_Data',
           'image_size': [480, 480],
           'n_frames': 6,
           'inwidth': 4096,
           'inheight': 3072,
           'data_suffix': '*.BGGR',
           'black_level': 1024,
           'white_level': 16383,
           'target_pattern': 'bggr',
           'input_pattern': 'bggr',
           'use_flip': True,
           'use_rot': True,
           }
    data = RawAlignDenoiseDataset(opt)
    data.__getitem__(0)

    #base = cv2.imread("test.png")
    # img = cv2.imread("test.png")
    #
    # img_motion = global_shift([img, img.copy()], 2)
    # #img_motion = local_motion([img, img.copy()], 2, 7, ch=3)
    # print(len(img_motion))
    # cv2.imwrite('base.png', img_motion[0])
    # cv2.imwrite('img_motion.png', img_motion[1])


