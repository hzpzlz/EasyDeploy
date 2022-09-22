# EasyDeploy Project
> To reduce deployment costs and speed up project schedules
## CSDN https://blog.csdn.net/hzpzlz5211314/article/details/126888166?spm=1001.2014.3001.5502

### Base On The Following Framework
>+ Opencv: 4.5.3 https://github.com/opencv/opencv 
>+ MNN: 2.1.0 https://github.com/alibaba/MNN      
>+ NCNN: Releases 20220729 https://github.com/Tencent/ncnn
>+ YAML-CPP: 0.7.0 https://github.com/jbeder/yaml-cpp/tree/yaml-cpp-0.7.0

## Set Network Params in assets/YourProjectName/xx.yaml 
> like the flowing example
```
project: squeezenet    #project name now notuse
model_path: ../assets/squeezenet/squeezenet_v1.1   #set model path without suffix

model_io:     #fixed
  input_info:   #fixed
    data:                 #your network first input name
      1 227 227 3 32      #n h w c datatype
    input_example1:       #your network second input name if exists
      1 666 666 3 32
  output_info:   #fixed   
    prob:                 #your network first output name
      1 1 1 1000 32       #n h w c datatype
    output_example1:      #your network second input name if exists
      1 666 666 1 32
```

## Run On Linux Platform
> support MNN and NCNN, change param in auto_run.sh
```
set 
ARCH='linux'
HPC_BACKEND='MNN'
DEMO_SELECT='segmentation'

sh auto_run.sh
```

## Run On Android Platform
> now only support MNN, change param in auto_run.sh
```
set 
ARCH='android'
HPC_BACKEND='MNN'
DEMO_SELECT='segmentation'  

sh auto_run.sh
```

## To Do Lists
>1. split demo and inferengine

## [Update log]

>+ 2022-09-22 finish yaml-cpp get params, support squeezenet(MNN, NCNN, linux now) segmentation(MNN, linux now)
>+ 2022-09-20 use yaml-cpp get params
>+ 2022-09-20 support demo selct and add segmentation(now only MNN) demos
>+ 2022-09-20 add NCNN infer framework, now support MNN+linux/MNN+android/NCNN+linux
>+ 2022-09-19 add infer engine and image process, support MNN+linux and MNN+android

