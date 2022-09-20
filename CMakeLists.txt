cmake_minimum_required(VERSION 3.10.2)

project(easy_deploy)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fexceptions -frtti -fomit-frame-pointer -ffunction-sections -fdata-sections -ffast-math -Os -Wall")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -pthread  -fexceptions -frtti -fomit-frame-pointer -ffunction-sections -fdata-sections -ffast-math -pie -fPIE -Os -Wall")
SET(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -Wl --allow-multiple-definition --whole-archive -pie -fPIE -fuse-ld=gold")
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")

set(EASY_DEPLOY_DEMOS ${DEMO_SELECT} CACHE STRING "select easy deploy sdk")

if(ANDROID)
    set(ARCH ${ANDROID_ABI})
    message(STATUS "ANDROID")
elseif(APPLE)
    message(STATUS "APPLE")
elseif(WIN32)
    message(STATUS "WIN32")
elseif(UNIX)
    set(ARCH "x86_64")
    message(STATUS "UNIX")
else()
    message(FATAL_ERROR "OTHER")
endif()

if(HPC_BACKEND STREQUAL "MNN")
    message(STATUS "Using MNN Inference Framework")
    add_definitions(-D__MNN_INFER_FRAMEWORK__)
elseif(HPC_BACKEND STREQUAL "NCNN")
    message(STATUS "Using NCNN Inference Framework")
    add_definitions(-D__NCNN_INFER_FRAMEWORK__)
else ()
    message(FATAL_ERROR "Not support this backend: " ${HPC_BACKEND})
endif()

message(STATUS "\n=========================== include ===========================\n")

#图像前后处理、项目的头文件以及推理框架的头文件
set(HPC_CLASS_INC ${CMAKE_SOURCE_DIR}/infer_engine/${HPC_BACKEND}/)
set(HPC_INC ${CMAKE_SOURCE_DIR}/dependency/include/${HPC_BACKEND}/ ${HPC_CLASS_INC})
set(YAML_INC ${CMAKE_SOURCE_DIR}/dependency/include/YAML) 
set(EASY_DEPLOY_INC ${CMAKE_SOURCE_DIR}/include/)
set(IMAGE_PROCESS_INC ${CMAKE_SOURCE_DIR}/image/include/)
#opencv头文件
set(OPENCV_INC ${CMAKE_SOURCE_DIR}/dependency/include/opencv4/)

message(STATUS "HPC_BACKEND_INC = " ${HPC_INC})
message(STATUS "YAML_INC = " ${YAML_INC})
message(STATUS "OPENCV_INC = " ${OPENCV_INC})
message(STATUS "IMAGE_PROCESS_INC = " ${IMAGE_PROCESS_INC})
message(STATUS "EASY_DEPLOY_INC = " ${EASY_DEPLOY_INC})

#include头文件
include_directories(
    ${OPENCV_INC}
    ${HPC_INC}
    ${YAML_INC}
    ${IMAGE_PROCESS_INC}
    ${EASY_DEPLOY_INC}
    )

message(STATUS "\n=========================== libs ===========================\n")

#推理框架的依赖库
if(HPC_BACKEND STREQUAL "MNN")
    set(HPC_LIB ${CMAKE_SOURCE_DIR}/dependency/libs/${ARCH}/${HPC_BACKEND}/libMNN.so)
elseif(HPC_BACKEND STREQUAL "NCNN")
    set(HPC_LIB ${CMAKE_SOURCE_DIR}/dependency/libs/${ARCH}/${HPC_BACKEND}/libncnn.so.1)
endif()

##yaml-cpp 依赖库
set(YAML_LIB ${CMAKE_SOURCE_DIR}/dependency/libs/${ARCH}/YAML/libyaml-cpp.a)

#opencv依赖库 多个.so文件，用FILE 
FILE(GLOB_RECURSE OPENCV_LIB ${CMAKE_SOURCE_DIR}/dependency/libs/${ARCH}/opencv2/*.so)

# message(STATUS "HPC_BACKEND_LIBS = ${HPC_LIB}")
# message(STATUS "YAML_LIB = ${YAML_LIB}")
# message(STATUS "OPENCV_LIB = ${OPENCV_LIB}")

set(EASY_DEPLOY_LIBS ${HPC_LIB} ${YAML_LIB} ${OPENCV_LIB})
message(STATUS "EASY_DEPLOY_LIBS = ${EASY_DEPLOY_LIBS}")

###适合安装好的的linux
# find_package(OpenCV)
# include_directories(${OpenCV_INCLUDE_DIRS})

# message(STATUS "OPENCV_INC = ${OpenCV_INCLUDE_DIRS}")
# message(STATUS "OPENCV_LIB = ${OpenCV_LIBRARIES}")

#前后处理和推理框架所需的cpp
FILE(GLOB_RECURSE hpc_class_cpp ${CMAKE_SOURCE_DIR}/infer_engine/${HPC_BACKEND}/*.cpp)
FILE(GLOB_RECURSE easy_deploy_cpp ${CMAKE_SOURCE_DIR}/src/*.cpp)
FILE(GLOB_RECURSE image_process_cpp ${CMAKE_SOURCE_DIR}/image/src/*.cpp)

set(EASY_DEPLOY_CPPS ${hpc_class_cpp} ${easy_deploy_cpp} ${image_process_cpp})

add_subdirectory(sdks)