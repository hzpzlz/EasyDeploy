cmake_minimum_required(VERSION 3.10.2)

project(easy_deploy_demo)

set(SELECT_DEMOS "")
if ("all" IN_LIST EASY_DEPLOY_DEMOS)
    list(APPEND SELECT_DEMOS "mnist")
    list(APPEND SELECT_DEMOS "segmentation")
    list(APPEND SELECT_DEMOS "squeezenet")
else ()
    set(SELECT_DEMOS ${EASY_DEPLOY_DEMOS})
endif()

foreach (select_demo IN LISTS SELECT_DEMOS)
    message(STATUS "======= build sdk: ${select_demo} =======\n")
    add_subdirectory(${select_demo})
endforeach()