cmake_minimum_required(VERSION 3.10.2)

project(segment)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fexceptions -frtti -fomit-frame-pointer -ffunction-sections -fdata-sections -ffast-math -Os -Wall")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -pthread  -fexceptions -frtti -fomit-frame-pointer -ffunction-sections -fdata-sections -ffast-math -pie -fPIE -Os -Wall")
SET(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -Wl --allow-multiple-definition --whole-archive -pie -fPIE -fuse-ld=gold")
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")

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
set(HPC_INC ${CMAKE_SOURCE_DIR}/dependency/inc/${HPC_BACKEND}/ ${HPC_CLASS_INC})
set(EASY_DEPLOY_INC ${CMAKE_SOURCE_DIR}/inc/)
set(IMAGE_PROCESS_INC ${CMAKE_SOURCE_DIR}/image/inc/)
#opencv头文件
set(OPENCV_INC ${CMAKE_SOURCE_DIR}/dependency/inc/opencv4/)

message(STATUS "HPC_BACKEND_INC = " ${HPC_INC})
message(STATUS "OPENCV_INC = " ${OPENCV_INC})
message(STATUS "IMAGE_PROCESS_INC = " ${IMAGE_PROCESS_INC})
message(STATUS "EASY_DEPLOY_INC = " ${EASY_DEPLOY_INC})

#include头文件
include_directories(${OPENCV_INC})
include_directories(${HPC_INC})
include_directories(${EASY_DEPLOY_INC})
include_directories(${IMAGE_PROCESS_INC})

message(STATUS "\n=========================== libs ===========================\n")

#推理框架的依赖库
if(HPC_BACKEND STREQUAL "MNN")
    set(HPC_LIB ${CMAKE_SOURCE_DIR}/dependency/libs/${ARCH}/${HPC_BACKEND}/libMNN.so)
elseif(HPC_BACKEND STREQUAL "NCNN")
    set(HPC_LIB ${CMAKE_SOURCE_DIR}/dependency/libs/${ARCH}/${HPC_BACKEND}/libncnn.so.1)
endif()

#opencv依赖库 多个.so文件，用FILE 
FILE(GLOB_RECURSE OPENCV_LIB ${CMAKE_SOURCE_DIR}/dependency/libs/${ARCH}/opencv2/*.so)

message(STATUS "HPC_BACKEND_LIBS = " ${HPC_LIB})
message(STATUS "OPENCV_LIB = ${OPENCV_LIB}")

###适合安装好的的linux
# find_package(OpenCV)
# include_directories(${OpenCV_INCLUDE_DIRS})

# message(STATUS "OPENCV_INC = ${OpenCV_INCLUDE_DIRS}")
# message(STATUS "OPENCV_LIB = ${OpenCV_LIBRARIES}")

#demo和前后处理所需的cpp
FILE(GLOB_RECURSE hpc_class_cpp ${CMAKE_SOURCE_DIR}/infer_engine/${HPC_BACKEND}/*.cpp)
FILE(GLOB_RECURSE easy_deploy_cpp ${CMAKE_SOURCE_DIR}/src/*.cpp)
FILE(GLOB_RECURSE image_process_cpp ${CMAKE_SOURCE_DIR}/image/src/*.cpp)
FILE(GLOB_RECURSE demo_cpp ${CMAKE_SOURCE_DIR}/demo/suqeezenet.cpp)
message(STATUS "DEMO_CPP = " ${demo_cpp})

message(STATUS "\n================== execute CMakeLists Done, Start Install =============\n")

#动态库
add_library(
    squeezenetlib
    SHARED
    ${hpc_class_cpp}
    ${easy_deploy_cpp}
    ${image_process_cpp}
    ${demo_cpp}
)
#链接所需的lib
target_link_libraries(
    squeezenetlib
    ${HPC_LIB}
    ${OPENCV_LIB}
)

# #静态库
# add_library(
#     seglibstatic
#     STATIC
#     ${hpc_class_cpp}
#     ${image_process_cpp}
#     ${easy_deploy_cpp}
#     ${demo_cpp}
# )
# #链接所需的lib
# target_link_libraries(
#     seglibstatic
#     ${HPC_LIB}
#     ${OPENCV_LIB}
# )

#生成可执行文件
add_executable(
    squeezenetDemo
    ${hpc_class_cpp}
    ${image_process_cpp}
    ${easy_deploy_cpp}
    ${demo_cpp}
)
#链接所需的lib
target_link_libraries(
    squeezenetDemo
    ${HPC_LIB}
    ${OPENCV_LIB}
)

##install
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build/install)
set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX}/bin/)
set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib/)
#set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/inc/)
set(path_demo_output ${CMAKE_INSTALL_PREFIX}/demo/)

install(
    TARGETS squeezenetDemo squeezenetlib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
#install(DIRECTORY ${path_inc_output} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${seg_demo_cpp} DESTINATION ${path_demo_output})