cmake_minimum_required(VERSION 3.10.2)

project(mnist)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src ${PROJECT_NAME}_cpps)
FILE(GLOB_RECURSE mnist_demo_cpp ${CMAKE_CURRENT_SOURCE_DIR}/demo/mnist.cpp)

#动态库
add_library(
    ${PROJECT_NAME}
    SHARED
    ${hpc_class_cpp}
    ${image_process_cpp}
    ${easy_deploy_cpp}
    ${mnist_demo_cpp}
)
#链接所需的lib
target_link_libraries(
    ${PROJECT_NAME}
    ${HPC_LIB}
    ${OPENCV_LIB}
)

#生成可执行文件
add_executable(
    ${PROJECT_NAME}Demo
    ${hpc_class_cpp}
    ${image_process_cpp}
    ${easy_deploy_cpp}
    ${mnist_demo_cpp}
)
#链接所需的lib
target_link_libraries(
    ${PROJECT_NAME}Demo
    ${HPC_LIB}
    ${OPENCV_LIB}
)

##install
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build/sdks/${PROJECT_NAME}/install)
set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX}/bin/)
set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib/)
#set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/inc/)

set(path_demo_install ${CMAKE_INSTALL_PREFIX}/demo/)
set(demo ${CMAKE_CURRENT_SOURCE_DIR}/demo)

install(
    TARGETS ${PROJECT_NAME}Demo ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
#install(DIRECTORY ${path_inc_output} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${demo} DESTINATION ${path_demo_install})

