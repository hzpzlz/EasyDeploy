# Install script for directory: /home/disk/4T/codes/Deploy/EasyDeploy/sdks/squeezenet

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/bin/squeezenetDemo" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/bin/squeezenetDemo")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/bin/squeezenetDemo"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/bin/squeezenetDemo")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/bin" TYPE EXECUTABLE FILES "/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/squeezenetDemo")
  if(EXISTS "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/bin/squeezenetDemo" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/bin/squeezenetDemo")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/bin/squeezenetDemo"
         OLD_RPATH "/home/disk/4T/codes/Deploy/EasyDeploy/dependency/libs/x86_64/NCNN:/home/disk/4T/codes/Deploy/EasyDeploy/dependency/libs/x86_64/opencv2:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/bin/squeezenetDemo")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/lib/libsqueezenet.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/lib/libsqueezenet.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/lib/libsqueezenet.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/lib/libsqueezenet.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/lib" TYPE SHARED_LIBRARY FILES "/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/libsqueezenet.so")
  if(EXISTS "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/lib/libsqueezenet.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/lib/libsqueezenet.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/lib/libsqueezenet.so"
         OLD_RPATH "/home/disk/4T/codes/Deploy/EasyDeploy/dependency/libs/x86_64/NCNN:/home/disk/4T/codes/Deploy/EasyDeploy/dependency/libs/x86_64/opencv2:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/lib/libsqueezenet.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/demo//demo")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/disk/4T/codes/Deploy/EasyDeploy/build/sdks/squeezenet/install/demo/" TYPE DIRECTORY FILES "/home/disk/4T/codes/Deploy/EasyDeploy/sdks/squeezenet/demo")
endif()

