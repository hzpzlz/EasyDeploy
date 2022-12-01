import os
import os.path
import cv2
import numpy as np
import torch
from torchvision import transforms
import data.noise_davis as noise_davis
import pandas as pd
import glob
from PIL import Image

class ImageDAVIS(torch.utils.data.Dataset):
    #def __init__(self, data_path, datatype="train", patch_size=None, stride=40):
    def __init__(self, opt):
        super(ImageDAVIS, self).__init__()
        self.data_path = opt['data_root']
        self.datatype = opt['phase']
        self.size = opt['image_size'] if self.datatype=='train' else None
        self.stride = opt['stride'] if self.datatype=='train' else None
        self.noise_dist = opt['noise_dist']
        self.noise_mode = opt['noise_mode']
        self.noise_std = opt['noise_std']

        self.min_noise = opt['min_noise'] if self.datatype=='train' else None
        self.max_noise = opt['max_noise'] if self.datatype=='train' else None

        if self.datatype == "train":
            self.folders = pd.read_csv(os.path.join(self.data_path, "ImageSets", "2017", "train.txt"), header=None)
        elif self.datatype == "val":
            self.folders = pd.read_csv(os.path.join(self.data_path, "ImageSets", "2017", "val.txt"), header=None)
        else:
            self.folders = pd.read_csv(os.path.join(self.data_path, "ImageSets", "2017", "test-dev.txt"), header=None)
        self.len = 0
        self.bounds = []

        for folder in self.folders.values:
            files = sorted(glob.glob(os.path.join(self.data_path, "JPEGImages", "480p", folder[0], "*.jpg")))
            self.len += len(files)
            self.bounds.append(self.len)

        if self.size is not None:
            self.n_H = (int((480-self.size)/self.stride)+1)
            self.n_W = (int((854-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches

        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders.values[i][0]
                if i>0:
                    index -= self.bounds[i-1]
                break

        files = sorted(glob.glob(os.path.join(self.data_path, "JPEGImages", "480p", folder, "*.jpg")))

        Img = np.array(Image.open(files[index]))

        if self.size is not None:
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]

        #if self.datatype=='train':
        #    noise = noise_davis.get_noise(Img, dist=self.noise_dist, mode=self.noise_mode, min_noise=self.min_noise, max_noise=self.max_noise, noise_std=self.noise_std)
        #elif self.datatype=='val':
        #    noise = noise_davis.get_noise(Img, dist=self.noise_dist, mode=self.noise_mode, noise_std=self.noise_std)
        #else:
        #    noise = noise_davis.get_noise(Img, dist=self.noise_dist, mode=self.noise_mode, noise_std=self.noise_std)
        #print(self.size, Img.shape, "iiiiiiiiiiiiiiiiiii")

        return self.transform(Img).type(torch.FloatTensor) #+noise

class DAVIS(torch.utils.data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_path = opt['data_root']
        self.datatype = opt['phase']
        self.size = opt['image_size'] if self.datatype=='train' else None
        self.stride = opt['stride'] if self.datatype=='train' else None
        self.noise_dist = opt['noise_dist']
        self.noise_mode = opt['noise_mode']
        self.noise_std = opt['noise_std']
        self.n_frames = opt['n_frames']

        if self.datatype == "train":
            self.folders = pd.read_csv(os.path.join(self.data_path, "ImageSets", "2017", "train.txt"), header=None)
        elif self.datatype == "val":
            self.folders = pd.read_csv(os.path.join(self.data_path, "ImageSets", "2017", "val.txt"), header=None)
        else:
            self.folders = pd.read_csv(os.path.join(self.data_path, "ImageSets", "2017", "test-dev.txt"), header=None)
        self.len = 0
        self.bounds = []

        for folder in self.folders.values:
            files = sorted(glob.glob(os.path.join(self.data_path, "JPEGImages", "480p", folder[0], "*.jpg")))
            self.len += len(files)
            self.bounds.append(self.len)

        if self.size is not None:
            self.n_H = (int((480-self.size)/self.stride)+1)
            self.n_W = (int((854-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches

        ends = 0
        x = (self.n_frames-1) // 2
        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders.values[i][0]
                if i>0:
                    index -= self.bounds[i-1]
                    newbound = bound - self.bounds[i-1]
                else:
                    newbound = bound
                if(index < x):
                    ends = x-index
                elif(newbound-1-index < x):
                    ends = -(x-(newbound-1-index))
                break

        files = sorted(glob.glob(os.path.join(self.data_path, "JPEGImages", "480p", folder, "*.jpg")))
        path_img = files[index]

        Img = Image.open(files[index])
        Img = np.array(Img)

        for i in range(1,x+1):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = Image.open(files[index-i+off])
            img = np.array(img)
            Img = np.concatenate((img, Img), axis=2)

        for i in range(1,x+1):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = Image.open(files[index+i-off])
            img = np.array(img)
            Img = np.concatenate((Img, img), axis=2)

        if self.size is not None:
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]

        return {'img_path' : path_img,  'data': self.transform(np.array(Img)).type(torch.FloatTensor)}
