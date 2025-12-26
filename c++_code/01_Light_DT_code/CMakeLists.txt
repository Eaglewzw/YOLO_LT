cmake_minimum_required(VERSION 3.10)

project(Light_DT)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)



# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
include_directories(/opt/TensorRT-8.6/include/)
link_directories(/opt/TensorRT-8.6/lib/)

link_directories(${PROJECT_SOURCE_DIR}/lib/)

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)
include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(${PROJECT_SOURCE_DIR}/include/ncnn/)
include_directories(${PROJECT_SOURCE_DIR}/include/MotionDetector/)
include_directories(${PROJECT_SOURCE_DIR}/include/Kalman/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)




find_package(OpenCV 4.6.0)
include_directories(${OpenCV_INCLUDE_DIRS})
#打印 OpenCV 的一些变量值
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV library directories: ${OpenCV_LIB_DIR}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(Light_DT main.cpp ${SRCS})
target_link_libraries(Light_DT -lopencv_world libncnn.so -lvulkan)
target_link_libraries(Light_DT nvinfer)
target_link_libraries(Light_DT cudart)
target_link_libraries(Light_DT ${OpenCV_LIBS})





