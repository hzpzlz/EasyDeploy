# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake-3.12.2/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.12.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/disk/4T/codes/Deploy/EasyDeploy

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/disk/4T/codes/Deploy/EasyDeploy/build

# Include any dependencies generated for this target.
include CMakeFiles/seglib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/seglib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/seglib.dir/flags.make

CMakeFiles/seglib.dir/src/seg_demo.cpp.o: CMakeFiles/seglib.dir/flags.make
CMakeFiles/seglib.dir/src/seg_demo.cpp.o: ../src/seg_demo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/disk/4T/codes/Deploy/EasyDeploy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/seglib.dir/src/seg_demo.cpp.o"
	/home/hzp/NDK/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android24 --gcc-toolchain=/home/hzp/NDK/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/hzp/NDK/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/seglib.dir/src/seg_demo.cpp.o -c /home/disk/4T/codes/Deploy/EasyDeploy/src/seg_demo.cpp

CMakeFiles/seglib.dir/src/seg_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/seglib.dir/src/seg_demo.cpp.i"
	/home/hzp/NDK/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android24 --gcc-toolchain=/home/hzp/NDK/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/hzp/NDK/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/disk/4T/codes/Deploy/EasyDeploy/src/seg_demo.cpp > CMakeFiles/seglib.dir/src/seg_demo.cpp.i

CMakeFiles/seglib.dir/src/seg_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/seglib.dir/src/seg_demo.cpp.s"
	/home/hzp/NDK/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android24 --gcc-toolchain=/home/hzp/NDK/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/hzp/NDK/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/disk/4T/codes/Deploy/EasyDeploy/src/seg_demo.cpp -o CMakeFiles/seglib.dir/src/seg_demo.cpp.s

# Object files for target seglib
seglib_OBJECTS = \
"CMakeFiles/seglib.dir/src/seg_demo.cpp.o"

# External object files for target seglib
seglib_EXTERNAL_OBJECTS =

libseglib.so: CMakeFiles/seglib.dir/src/seg_demo.cpp.o
libseglib.so: CMakeFiles/seglib.dir/build.make
libseglib.so: ../dependency/libs/arm64-v8a/MNN/libMNN.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_calib3d.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_core.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_dnn.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_features2d.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_flann.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_gapi.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_highgui.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_imgcodecs.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_imgproc.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_ml.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_objdetect.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_photo.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_stitching.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_video.so
libseglib.so: ../dependency/libs/arm64-v8a/opencv2/libopencv_videoio.so
libseglib.so: CMakeFiles/seglib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/disk/4T/codes/Deploy/EasyDeploy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libseglib.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/seglib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/seglib.dir/build: libseglib.so

.PHONY : CMakeFiles/seglib.dir/build

CMakeFiles/seglib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/seglib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/seglib.dir/clean

CMakeFiles/seglib.dir/depend:
	cd /home/disk/4T/codes/Deploy/EasyDeploy/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/disk/4T/codes/Deploy/EasyDeploy /home/disk/4T/codes/Deploy/EasyDeploy /home/disk/4T/codes/Deploy/EasyDeploy/build /home/disk/4T/codes/Deploy/EasyDeploy/build /home/disk/4T/codes/Deploy/EasyDeploy/build/CMakeFiles/seglib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/seglib.dir/depend

