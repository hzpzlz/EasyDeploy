#!/bin/bash
#rootpath='/home/disk/4T/datasets/hdrdata/0531_data/0531_problems/p5'
resultpath='/home/disk/4T/codes/align_nightphoto/ai_super_night_alignment/data/input'
#resultpath='/home/disk/4T/datasets/hdrdata/1018_data/problems/p1_ttt'
#resultpath='/home/disk/4T/datasets/hdrdata/ground_H'
imgpath=$resultpath'_imgs'
ext='_new'
python show_flow.py --root_path $resultpath --flag $ext --ext '*.dat'
#python show_flow.py --root_path $resultpath --flag $ext --ext '*.sdat'

#python show_ideal.py --root_path $resultpath --ext '.alignraw' --flag 'ev-_resr10'$ext --ratio 1
python show_ideal.py --root_path $resultpath --ext '.craw' --flag 'ev0output'$ext --ratio 1 --oriraw true
#python show_ideal.py --root_path $resultpath --ext '.hres' --flag 'evneg'$ext --ratio 10 --oriraw true
#python show_ideal.py --root_path $resultpath --ext '.' --flag 'ev-inputratio'$ext --ratio 20 --oriraw true
#python show_ideal.py --root_path $resultpath --ext '.denoiseraw' --flag 'ev-denoised_ori'$ext --ratio 1
#python show_ideal.py --root_path $resultpath --ext '.denoiseraw' --flag 'ev-denoised_ratio10'$ext --ratio 10
#python show_ideal.py --root_path $resultpath --ext '.algideal' --flag 'ev0'$ext --ratio 1
#python show_ideal.py --root_path $resultpath --ext '.BGGR' --flag 'oridata'$ext
#python show_ideal.py --root_path $resultpath --ext '.forMevRaw' --flag 'aligninput'$ext --ratio 10
#python show_ideal.py --root_path $resultpath --ext '.forEV0Raw' --flag 'aligninput'$ext --ratio 1
#python show_ideal.py --root_path $resultpath --ext '.deghostraw' --flag 'deghost_'$ext
#python read_deghost.py --root_path $resultpath --ext '*.deghostmask' --flag '_ev-_fianlmask'$ext
#python read_deghost.py --root_path $resultpath --ext '*.cluma' --flag '_ev0'$ext
#python read_deghost.py --root_path $resultpath --ext '*.gray' --flag '_gray'$ext
python img_move.py --root_path $resultpath --result_path $imgpath --flag $ext

#python isp_move.py --root_path $resultpath --result_path $imgpath --flag $ext
