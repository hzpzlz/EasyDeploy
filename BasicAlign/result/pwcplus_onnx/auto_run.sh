#!/bin/bash
adb root
adb remount

root=/data/local/optflow
dlc_path=/home/hzp/codes/BasicAlign/result/pwcplus_onnx/pwcplus_95000_G.dlc

#建立文件夹
adb shell mkdir -p $root
adb shell mkdir -p $root/arm64
adb shell mkdir -p $root/arm64/lib
adb shell mkdir -p $root/arm64/bin
adb shell mkdir -p $root/output

#push 依赖
adb push $SNPE_ROOT/lib/aarch64-android-clang8.0/libSNPE.so $root/arm64/lib
adb push $SNPE_ROOT/lib/aarch64-android-clang8.0/libc++_shared.so $root/arm64/lib
adb push $SNPE_ROOT/bin/aarch64-android-clang8.0/snpe-net-run $root/arm64/bin

#push 数据
adb push rawdata.txt $root
adb push rawdata_bin $root

#push dlc
adb push $dlc_path $root

#push shell
adb push run_in_phone.sh $root

adb shell < run_in_phone.sh

#pull to linux
adb pull $root/output ./

#手机里设置环境变量
#adb shell export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/optflow/arm64/lib
#adb shell export PATH=$PATH:/data/local/optflow/arm64/bin

#snpe version
#adb shell snpe-net-run --version

#run
#snpe-net-run --container pwcplus_95000_G.dlc --input_list rawdata.txt --output_dir output
#adb shell < run.sh
