# EasyDeploy Project
## To reduce deployment costs and speed up project schedules
## CSDN https://blog.csdn.net/hzpzlz5211314/article/details/126888166?spm=1001.2014.3001.5502

### base on the following framework
>+ Opencv: 4.5.3
>+ MNN: 2.1.0
>+ NCNN: Releases 20220729

## run on linux
> support MNN and NCNN
```
set ARCH='linux' and HPC_BACKEND='MNN' or 'NCNN'

sh auto_run.sh
```

## run on android
> now only support MNN
```
set ARCH='android' and HPC_BACKEND='MNN'

sh auto_run.sh
```

## To Do Lists
>1. set param and read param auto
>2. split demo and inferengine
>3. ...

## [Update log]

2022-09-20 add NCNN infer framework, now support MNN+linux/MNN+android/NCNN+linux
2022-09-19 add infer engine and image process, support MNN+linux and MNN+android

