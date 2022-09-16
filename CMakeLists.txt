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

#头文件
if(HPC_BACKEND STREQUAL "MNN")
    set(HPC_INC ${CMAKE_SOURCE_DIR}/dependency/inc/${HPC_BACKEND}/)
elseif(HPC_BACKEND STREQUAL "NCNN")
    set(HPC_INC ${CMAKE_SOURCE_DIR}/dependency/inc/${HPC_BACKEND}/)
endif()

#依赖库
if(HPC_BACKEND STREQUAL "MNN")
    set(HPC_LIB ${CMAKE_SOURCE_DIR}/dependency/libs/${ARCH}/${HPC_BACKEND}/libMNN.so)
    # set(HPC_LIB ${CMAKE_SOURCE_DIR}/dependency/libs/${ARCH}/${HPC_BACKEND})
    # LINK_DIRECTORIES(${HPC_LIB})
elseif(HPC_BACKEND STREQUAL "NCNN")
    set(HPC_LIB ${CMAKE_SOURCE_DIR}/dependency/libs/${ARCH}/${HPC_BACKEND}/libNCNN.so)
endif()

message(STATUS "HPC_BACKEND_INC = " ${HPC_INC})
message(STATUS "HPC_BACKEND_LIBS = " ${HPC_LIB})

##opencv 头文件
set(OPENCV_INC ${CMAKE_SOURCE_DIR}/dependency/inc/opencv4/)
##opencv 依赖库 多个.so文件，用FILE 
FILE(GLOB_RECURSE OPENCV_LIB ${CMAKE_SOURCE_DIR}/dependency/libs/${ARCH}/opencv2/*.so)

message(STATUS "OPENCV_INC = ${OPENCV_INC}")
message(STATUS "OPENCV_LIB = ${OPENCV_LIB}")

###适合安装好的的linux
# find_package(OpenCV)
# include_directories(${OpenCV_INCLUDE_DIRS})

# message(STATUS "OPENCV_INC = ${OpenCV_INCLUDE_DIRS}")
# message(STATUS "OPENCV_LIB = ${OpenCV_LIBRARIES}")

#include头文件
include_directories(${OPENCV_INC})
include_directories(${HPC_INC})

#编写的demo和前后处理所需的头文件以及Cpp
FILE(GLOB_RECURSE mnist_demo_cpp ${CMAKE_SOURCE_DIR}/src/mnist_demo.cpp)
FILE(GLOB_RECURSE seg_demo_cpp ${CMAKE_SOURCE_DIR}/src/seg_demo.cpp)

message(STATUS "DEMO_CPP = " ${seg_demo_cpp})

#动态库
add_library(
    seglib
    SHARED
    ${seg_demo_cpp}
)
#链接所需的lib
target_link_libraries(
    seglib
    ${HPC_LIB}
    ${OPENCV_LIB}
)

# #静态库
# add_library(
#     seglibstatic
#     STATIC
#     ${seg_demo_cpp}
# )
# #链接所需的lib
# target_link_libraries(
#     seglibstatic
#     ${HPC_LIB}
#     ${OPENCV_LIB}
# )

#生成可执行文件
add_executable(
    segDemo
    ${seg_demo_cpp}
)
#链接所需的lib
target_link_libraries(
    segDemo
    ${HPC_LIB}
    ${OPENCV_LIB}
)

##install
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build/install)
set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX}/bin/)
set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib/)
# set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/inc/)
set(path_demo_output ${CMAKE_INSTALL_PREFIX}/demo/)

install(
    TARGETS segDemo seglib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# install(DIRECTORY ${path_inc_output} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${seg_demo_cpp} DESTINATION ${path_demo_output})