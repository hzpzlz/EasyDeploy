#!/bin/bash
####目前仅使用于linux和phone的CPU后端，后续会更新DSP和GPU

ARCH='android'  # "linux" "android"

mkdir build
cd build
rm * -rf
../build.sh $ARCH

make -j8
make install

###for linux
if [ "${ARCH}" = "linux" ]; then
    echo "linux ********"
    ./install/bin/segDemo
elif [ "${ARCH}" = "android" ]; then
    echo "android ******* "
    adb root
    adb remount
    root=/data/local/seg_hzp
    #建立文件夹
    adb shell mkdir -p $root
    adb shell mkdir -p $root/assets
    adb shell mkdir -p $root/lib
    adb shell mkdir -p $root/bin

    project_path=/home/disk/4T/codes/Deploy/EasyDeploy
    
    #push 数据以及模型
    adb push $project_path/assets/1.bmp $root/assets/
    adb push $project_path/assets/seg.mnn $root/assets/

    adb push $project_path/build/segDemo $root/bin/
    adb push $project_path/build/*.so $root/lib/
    #目前是使用opencv和MNN动态库编译而来的，所以还得把相应的.so push进去
    adb push $project_path/dependency/libs/arm64-v8a/MNN/*.so $root/lib/
    adb push $project_path/dependency/libs/arm64-v8a/opencv2/*.so $root/lib/

    #export 环境
    adb shell "
    cd ${root}
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${root}/lib
    cd bin
    ./segDemo
    "
    adb pull $root/assets/seg.jpg $project_path/assets/phone_res/
fi