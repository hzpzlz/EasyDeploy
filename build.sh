ARCH=$1    # suport  'linux' 'android'
ABI='arm64-v8a'   # only support arm68-v8a 
HPC_BACKEND=$2 # now only support MNN, support NCNN soon
DEMO_SELECT=$3    #'all' means compile all project; include 'segmentation' 'mnist' 'squeezenet'

#!/bin/bash

if ! [ -z ${NDKROOT} ]; then
    echo "NDKROOT: ${NDKROOT} was defined"
else
    echo "NDKROOT need to be set $NDKROOT"
fi

if [ "$ARCH" = "linux" ]; then
    echo "---------------------cmake linux ---------------------"
    cmake -DARCH=$ARCH -DHPC_BACKEND=$HPC_BACKEND -DDEMO_SELECT=${DEMO_SELECT} ../
elif [ "$ARCH" = "android" ]; then
    echo "---------------------cmake android ---------------------"
    cmake -DDEBUG=NO -DCMAKE_TOOLCHAIN_FILE=$NDKROOT/build/cmake/android.toolchain.cmake \
            -DANDROID_NDK=$NDKROOT \
            -DANDROID_ABI=${ABI} \
            -DANDROID_PLATFORM=android-24 \
            -DANDROID_TOOLCHAIN_NAME=clang \
            -DANDROID_STL=c++_shared \
            -DARCH=$ARCH \
            -DHPC_BACKEND=$HPC_BACKEND \
            ../
else    
    echo "not support $ARCH now, please check!!!"
fi
