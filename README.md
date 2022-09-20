# EasyDeploy Project
> To reduce deployment costs and speed up project schedules
## CSDN https://blog.csdn.net/hzpzlz5211314/article/details/126888166?spm=1001.2014.3001.5502

### Base On The Following Framework
>+ Opencv: 4.5.3 https://github.com/opencv/opencv 
>+ MNN: 2.1.0 https://github.com/alibaba/MNN      
>+ NCNN: Releases 20220729 https://github.com/Tencent/ncnn
>+ YAML-CPP: 0.7.0 https://github.com/jbeder/yaml-cpp/tree/yaml-cpp-0.7.0

## Run On Linux Platform
> support MNN and NCNN, change param in auto_run.sh
```
set ARCH='linux' and HPC_BACKEND='MNN'
set DEMO_SELECT='segmentation'

sh auto_run.sh
```

## Run On Android Platform
> now only support MNN, change param in auto_run.sh
```
set ARCH='android' and HPC_BACKEND='MNN'
set DEMO_SELECT='segmentation'

sh auto_run.sh
```

## To Do Lists
>1. set param and read param auto
>2. split demo and inferengine
>3. support option build demo

## [Update log]

>+ 2022-09-20 use yaml-cpp get params
>+ 2022-09-20 support demo selct and add segmentation(now only MNN) demos
>+ 2022-09-20 add NCNN infer framework, now support MNN+linux/MNN+android/NCNN+linux
>+ 2022-09-19 add infer engine and image process, support MNN+linux and MNN+android

