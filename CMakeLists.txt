cmake_minimum_required(VERSION 3.10)

project(AI)

if(NOT DEFINED deployment_dir)
    set(deployment_dir ${CMAKE_CURRENT_SOURCE_DIR})
endif(NOT DEFINED deployment_dir)

file(GLOB SRCFILES ${deployment_dir}/src/*.cpp)
file(GLOB APIFILES ${deployment_dir}/src/api/*.cpp)
file(GLOB NETFILES ${deployment_dir}/src/net/*.cpp)
file(GLOB COREFILES ${deployment_dir}/src/core/*.cpp)

file(GLOB CUDAFILES ${deployment_dir}/src/net/*.cu)

set(LIBS_DIR ${deployment_dir}/../data/libs)

find_package(CUDA)
if(CUDA_FOUND)

    find_library(NVINFER NAMES nvinfer REQUIRED
    HINTS ${LIBS_DIR}/TensorRT-8.4.1.5.Windows10.x86_64.cuda-11.6.cudnn8.4/lib)

    add_definitions(-DCUDA_FOUND)
    include_directories($ENV{CUDA_PATH}/include)

    if(WIN32)

    link_directories(${LIBS_DIR}/TensorRT-8.4.1.5.Windows10.x86_64.cuda-11.6.cudnn8.4/lib)
    include_directories(${LIBS_DIR}/TensorRT-8.4.1.5.Windows10.x86_64.cuda-11.6.cudnn8.4/include)
    
    set(ONNX ${LIBS_DIR}/onnxruntime-win-x64-gpu-1.11.1)
    set(TORCH ${LIBS_DIR}/libtorch-win-shared-with-deps-debug-1.12.0+cu116)
    
    elseif(UNIX)

    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
    
    set(ONNX ${LIBS_DIR}/onnxruntime-linux-x64-gpu-1.11.1)
    set(TORCH ${LIBS_DIR}/libtorch-cxx11-abi-shared-with-deps-1.12.0+cu116)

    endif()
else(CUDA_FOUND)
    if(WIN32)

    set(ONNX ${LIBS_DIR}/onnxruntime-win-x64-1.11.1)
    set(TORCH ${LIBS_DIR}/libtorch-win-shared-with-deps-debug-1.12.0+cpu)

    elseif(UNIX)
    
    set(ONNX ${LIBS_DIR}/onnxruntime-linux-x64-1.11.1)
    set(TORCH ${LIBS_DIR}/libtorch-cxx11-abi-shared-with-deps-1.12.0+cpu)

    endif()
endif(CUDA_FOUND)

if(WIN32)
set(OpenCV_DIR ${LIBS_DIR}/opencv-4.6.0/build)
link_directories(${LIBS_DIR}/opencv-4.6.0/build/x64/vc15/bin)

set(NCNN ${LIBS_DIR}/ncnn-20220701-windows-vs2022-shared/x64)
set(Vulkan_INCLUDE_DIR "C:\\VulkanSDK\\1.3.216.0\\Include\\")
set(Vulkan_LIBRARY "C:\\VulkanSDK\\1.3.216.0\\Lib\\")

set(CUDNN_LIBRARY_PATH "")
set(CUDNN_INCLUDE_PATH ${LIBS_DIR}/cudnn-windows-x86_64-8.4.1.50_cuda11.6-archive/include)
list(APPEND CUDNN_LIBRARY_PATH ${LIBS_DIR}/cudnn-windows-x86_64-8.4.1.50_cuda11.6-archive/lib)
list(APPEND CUDNN_LIBRARY_PATH ${LIBS_DIR}/cudnn-windows-x86_64-8.4.1.50_cuda11.6-archive/bin)

elseif(UNIX)
set(NCNN ${LIBS_DIR}/ncnn-20220701-ubuntu-2004-shared)

find_package(ncnn)

endif()

set(ONNX_DIR ${ONNX})
set(ncnn_DIR ${NCNN}/lib/cmake/ncnn)
set(Torch_DIR ${TORCH}/share/cmake/Torch)


foreach(LIB_DIR
    ${NCNN} 
    ${ONNX}
    ${TORCH}
)
    file(TO_NATIVE_PATH ${LIB_DIR} LIB_DIR)
    link_directories(${LIB_DIR}/bin)
    link_directories(${LIB_DIR}/lib)
    include_directories(${LIB_DIR}/include)

endforeach(LIB_DIR)

find_package(OpenCV)
find_package(Torch)

find_library(ONNX_FOUND NAMES onnxruntime HINTS ${ONNX}/lib REQUIRED)

if(NVINFER)

cuda_add_library(
    DAAI 
    ${CUDAFILES}
    ${SRCFILES}
    ${COREFILES}
    ${APIFILES}
    ${NETFILES}
)
cuda_add_executable(AI ${deployment_dir}/src/daai.cpp)

add_definitions(-DTRT_FOUND)
target_link_libraries(DAAI nvinfer cudart)
else(NVINFER)
add_library(
    DAAI 
    ${SRCFILES}
    ${COREFILES}
    ${APIFILES}
    ${NETFILES}
)
add_executable(AI ${deployment_dir}/src/daai.cpp)
endif(NVINFER)

if(ncnn_FOUND)
add_definitions(-DNCNN_FOUND)
target_link_libraries(DAAI ncnn)
endif(ncnn_FOUND)

if(ONNX_FOUND)
add_definitions(-DONNX_FOUND)
target_link_libraries(DAAI onnxruntime)
endif(ONNX_FOUND)

if(Torch_FOUND)
add_definitions(-DTORCH_FOUND)
target_link_libraries(DAAI ${TORCH_LIBRARIES})
endif(Torch_FOUND)

if(OPENCV_FOUND)
add_definitions(-DOPENCV_FOUND)
target_link_libraries(DAAI ${OpenCV_LIBS})
include_directories(/usr/local/include/opencv4)
endif(OPENCV_FOUND)

if(NOT DEFINED ANDROID AND NOT DEFINED MSVC)
# TODO
target_link_libraries(DAAI pthread)
endif()

target_include_directories(DAAI PUBLIC ${deployment_dir}/src)

target_link_libraries(AI DAAI)
