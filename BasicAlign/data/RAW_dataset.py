import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import glob
import os
from .preproc.bayer import bayer_symmetry_random_crop
from .preproc.raw import pack_raw_to_4ch, rescaling, bayer_to_offsets
from .preproc.raw_io import load_plainraw
import data.util as util

def get_full_dirs(dir):
    dir_list = []
    batch_lists = os.listdir(dir)
    for i in batch_lists:
        scenes = os.listdir(os.path.join(dir, i))
        for j in scenes:
            dir_list.append(os.path.join(dir, i, j))
    return dir_list

class RawDataset(data.Dataset):
    def __init__(self, opt):
        super(RawDataset, self).__init__()
        self.opt = opt
        self.all_path = get_full_dirs(opt['dataroot'])
        #print(len(self.all_path), 'ffffffffffff')

    def __getitem__(self, index):
        base_path, ref_path = None, None
        img_size = self.opt['image_size']

        # get GT image
        data_dir = self.all_path[index]
        #data_lists = sorted(glob.glob(data_dir + self.opt['data_suffix']))
        
        base_path = glob.glob(os.path.join(data_dir, "IMG_2022*02_EV[0*.BGGR"))[0]
        

        if self.opt['phase'] == 'train':
            ref_random = np.random.randint(3, 8)
            ref_path = glob.glob(os.path.join(data_dir, "IMG_2022*0" + str(ref_random) + "_EV[0*.BGGR"))[0]
        else:
            ref_path = glob.glob(os.path.join(data_dir, "IMG_2022*07_EV[0*.BGGR"))[0]
        
        #base_img = load_plainraw(base_path, self.opt['inwidth'], self.opt['inheight'])
        #ref_img = load_plainraw(ref_path, self.opt['inwidth'], self.opt['inheight'])
        base_img = np.fromfile(base_path, dtype = np.uint16).reshape(self.opt['inheight'], self.opt['inwidth'])
        ref_img = np.fromfile(ref_path, dtype = np.uint16).reshape(self.opt['inheight'], self.opt['inwidth'])
        
        input_data = np.zeros((self.opt['inheight'], self.opt['inwidth'], 2))
        base_img = rescaling(base_img, self.opt['black_level'], self.opt['white_level'])
        base_img = np.clip(base_img, 0.0, 1.0)
        #input_data[:,:,0]=base_img
        
        ref_img = rescaling(ref_img, self.opt['black_level'], self.opt['white_level'])
        ref_img = np.clip(ref_img, 0.0, 1.0)
        #input_data[:,:,1]=ref_img
        
        offset = bayer_to_offsets(self.opt['target_pattern'])
        
        #if self.opt['phase'] == 'train':
        #    symmetry_flags = np.random.randint(2, size=3)
        #    input_data = bayer_symmetry_random_crop(input_data, self.opt['input_pattern'], self.opt['target_pattern'], symmetry_flags[0],symmetry_flags[1],symmetry_flags[2],img_size[0], img_size[1])
        
        #img_base = pack_raw_to_4ch(input_data[:,:,0], offset)
        #img_ref = pack_raw_to_4ch(input_data[:,:,1], offset)
        img_base = pack_raw_to_4ch(base_img, offset)
        img_ref = pack_raw_to_4ch(ref_img, offset)

        if self.opt['phase'] == 'train':
            H,W = img_base.shape[0:2]
            rnd_h = random.randint(0, max(0, H - img_size[0]))
            rnd_w = random.randint(0, max(0, W - img_size[1]))

            img_base = img_base[rnd_h:rnd_h+img_size[0], rnd_w:rnd_w+img_size[1], :]
            img_ref = img_ref[rnd_h:rnd_h+img_size[0], rnd_w:rnd_w+img_size[1], :]

            img_ref, img_base = util.augment([img_ref, img_base], self.opt['use_flip'], self.opt['use_rot'])

        img_base = torch.from_numpy(np.ascontiguousarray(np.transpose(img_base, (2, 0, 1)))).float()
        img_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(img_ref, (2, 0, 1)))).float()

        return {'base': img_base, 'ref': img_ref, 'base_path': base_path, 'ref_path': ref_path}

    def __len__(self):
        return len(self.all_path)

if __name__ == '__main__':
    opt = {'phase': 'train',
           'dataroot': '/home/work/ssd2/datasets/RawAlign/train_data',
           'image_size': [480, 480], 
           'inwidth': 4096, 
           'inheight': 3072,
           'data_suffix': '*.BGGR',
           'black_level': 1024,
           'white_level': 16383,
           'target_pattern': 'bggr',
           'input_pattern': 'bggr',
           }
    data = RawDataset(opt)
    data.__getitem__(0)
    #print(data.__len__())


