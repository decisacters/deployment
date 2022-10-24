
# install cmake from cmake.org
# 
# install g++ compiler and gdb debugger
# https://www.msys2.org/
# 
# pacman -S mingw-w64-x86_64-toolchain
# select mingw-w64-x86_64-gcc and mingw-w64-x86_64-gdb, mingw-w64-x86_64-make
# 
# install ninja 
# https://github.com/ninja-build/ninja/releases
# copy ninja to the C:\msys64\mingw64\bin
#
# install opencv
# https://github.com/opencv/opencv/releases
# compile using C:\msys64\mingw64\bin\g++.exe
# mkdir .\.github\opencv\bin
# cd .\.github\opencv
# cmake . -B bin -G "MinGW Makefiles" -W no-dev -D CMAKE_BUILD_TYPE=Debug -D CMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe -D CMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe -D OPENCV_VS_VERSIONINFO_SKIP=1 -D BUILD_LIST=core,imgproc,highgui #-D BUILD_opencv_world=ON

# install onnxruntime
# https://github.com/microsoft/onnxruntime/releases
# 
# install torch
#
# install CUDA
# copy content from cuda to vs
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\extras\visual_studio_integration\MSBuildExtensions
# C:\Program Files\Microsoft Visual Studio\2022\Community\Msbuild\Microsoft\VC\v170\BuildCustomizations
#
# install cudnn
# https://developer.nvidia.com/compute/cudnn/secure/8.4.1/local_installers/11.6/cudnn-windows-x86_64-8.4.1.50_cuda11.6-archive.zip
# https://developer.nvidia.com/compute/cudnn/secure/8.4.1/local_installers/11.6/cudnn-local-repo-ubuntu2004-8.4.1.50_1.0-1_amd64.deb
#
# install tensorrt
# must be same version as pytorch export and same gpu compute compatibility
# https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.4.1/zip/TensorRT-8.4.1.5.Windows10.x86_64.cuda-11.6.cudnn8.4.zip
# https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.4.1/local_repos/nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.1.5-ga-20220604_1-1_amd64.deb
# 
# install vulkan
# https://vulkan.lunarg.com/sdk/home#windows
# Vulkan_INCLUDE_DIR C:\VulkanSDK\1.3.216.0\Include
# Vulkan_LIBRARY C:\VulkanSDK\1.3.216.0\Lib
# 
# install ncnn
#