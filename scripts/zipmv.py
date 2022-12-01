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
args=parser.parse_args()

root_path = args.root_path
zip_lists = glob.glob(os.path.join(root_path, '*.zip'))
print(zip_lists)

for szip in zip_lists:
    fullname = szip.split('/')[-1]
    zipname = fullname.split('.')[0]
    respath = os.path.join(root_path, zipname)
    if not os.path.exists(respath):
        os.makedirs(respath)
 
    shutil.move(szip, os.path.join(respath, fullname))
    #unzip(os.path.join(respath, fullname))
#    out_dir = os.path.join(out_path, root_dir)
#    if not os.path.exists(out_dir):
#        os.makedirs(out_dir)
#    file_path = os.path.join(root_path, root_dir)
#
#    #old_img_list = sorted(glob.glob(os.path.join(file_path, '*_' + args.result_dir + '_*.raw')))
#    #old_img_list = sorted(glob.glob(os.path.join(file_path, '*.xml')))
#    for file in os.listdir(file_path):
#        #if file.endswith('.jpg') or file.endswith('.png'):
#        #    #print(file)
#        #    old_path = os.path.join(file_path, file)
#        #    new_path = os.path.join(out_dir, file)
#        #    shutil.copy(old_path, new_path)
#        if file.endswith('.txt') or file.endswith('.xml'):
#            #print(file)
#            old_path = os.path.join(file_path, file)
#            new_path = os.path.join(out_dir, file)
#            shutil.copy(old_path, new_path)
        #img_name = os.path.basename(img_path)
        #new_img_path = os.path.join(out_dir, img_name)

        
        #print(img_name, "****")
        #shutil.move(img_path, new_img_path)
        #shutil.copy(img_path, new_img_path)
