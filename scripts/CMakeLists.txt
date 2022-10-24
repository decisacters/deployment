cmake_minimum_required(VERSION 3.10)

project(daai)

add_definitions(-DSIMPLEOCV)

link_directories(${CMAKE_SOURCE_DIR})

add_executable(daai daai.cpp)

target_link_libraries(daai DAAI)
target_link_libraries(daai ncnn)
target_link_libraries(daai pthread)

cmake_minimum_required(VERSION 3.10)

project(AI)

set(deployment_dir ${CMAKE_CURRENT_SOURCE_DIR}/deployment)

include(${CMAKE_CURRENT_SOURCE_DIR}/deployment/CMakeLists.txt)
