#!/bin/bash
rootpath='/home/hzp/project/HDR/pipline/NightHDR/result/test_result'
#resultpath='/home/hzp/datasets/night_scan/L1_project/0223_4F/0223_4F_resimg'
#python show_ideal.py --root_path $rootpath
#python copy_result.py --root_path $rootpath --result_path $rootpath'img'
#python copy_result.py --root_path $rootpath --result_path $resultpath
#python align_mv.py --root_path $rootpath --result_path $rootpath'_alg'
python meta_mv.py --root_path $rootpath --result_path $rootpath'_metas'


