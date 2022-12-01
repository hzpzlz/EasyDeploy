#!/bin/bash
#source /home/hzp/anaconda3/bin/activate snpe1.53
#source $SNPE_ROOT/bin/envsetup.sh -t /home/hzp/anaconda3/envs/snpe1.53/lib/python3.6/site-packages/torch

echo $SNPE_ROOT
#onnx转dlc
input_onnx='/home/hzp/codes/BasicAlign/result/pwcplus_onnx/pwcplus_95000_G.onnx' 
output_dlc='/home/hzp/codes/BasicAlign/result/pwcplus_onnx/pwcplus_95000_G.dlc' 
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
    --input_network $input_onnx \
    --output_path $output_dlc
#
#$SNPE_ROOT/bin/x86_64-linux-clang/snpe-dlc-info --input_dlc $output_dlc

#dlc量化
output_quant='/home/hzp/codes/BasicAlign/result/pwcplus_onnx/pwcplus_95000_G_quant.dlc'
raw_data_path='/home/hzp/codes/BasicAlign/result/pwcplus_onnx/rawdata.txt' 
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-dlc-quantize \
    --input_dlc $output_dlc \
    --input_list $raw_data_path \
    --output_dlc $output_quant \
    --enable_htp

#res_output='/home/hzp/codes/BasicAlign/result/pwcplus_onnx/output' 
res_output='/home/hzp/codes/BasicAlign/result/pwcplus_onnx/output_quant' 
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-net-run \
    --container $output_quant \
    --input_list $raw_data_path \
    --output_dir $res_output
