#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import glob
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root_path',type=str,default='./result/../')
parser.add_argument('--ext', type=str, default='', help='')
args=parser.parse_args()

root_path = args.root_path
ext = args.ext

for root_dir in os.listdir(root_path): 
    if root_dir == 'result': continue
    print(root_dir)
    file_path = os.path.join(root_path, root_dir)
    #raw_txt = open(os.path.join(file_path, 'rawsFileName.txt'), 'w')
    if os.path.isdir(file_path):
        raw_txt = open(os.path.join(file_path, 'rawsFileName.txt'), 'w')
        img_list = sorted(glob.glob(os.path.join(file_path, '*.'+ext)))[0:8]
        for i in range(len(img_list)):
            raw_txt.write(img_list[i])
            raw_txt.write('\n')
        raw_txt.close()
