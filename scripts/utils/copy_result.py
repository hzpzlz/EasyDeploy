#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import glob
import shutil
import glob
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root_path',type=str,default='/home/hzp/datasets/night_scan/raw_base_align')
parser.add_argument('--result_dir',type=str,default='/home/hzp/datasets/night_scan/raw_base_align')
args=parser.parse_args()

root_path = args.root_path

out_path = os.path.join(os.path.dirname(root_path), args.result_dir)
if not os.path.exists(out_path):
    os.makedirs(out_path)

for root_dir in os.listdir(root_path): 
    out_dir = os.path.join(out_path, root_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    file_path = os.path.join(root_path, root_dir)

    old_img_list = sorted(glob.glob(os.path.join(file_path, '*.jpg')))
    for img_path in old_img_list:
        img_name = os.path.basename(img_path)
        new_img_path = os.path.join(out_dir, img_name)
        if(os.path.exists(new_img_path)):
            os.remove(new_img_path)
        shutil.move(img_path, new_img_path)
