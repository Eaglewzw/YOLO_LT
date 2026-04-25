cmake_minimum_required(VERSION 3.10)

project(Light_DT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Project headers
include_directories(${PROJECT_SOURCE_DIR}/include/)

# Source files
file(GLOB SRCS
    ${PROJECT_SOURCE_DIR}/src/yolo_detector.cpp
    ${PROJECT_SOURCE_DIR}/src/light_track.cpp
    ${PROJECT_SOURCE_DIR}/src/motion_detector.cpp
)

# OpenCV
find_package(OpenCV 4.6.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ONNX Runtime (system-installed)
find_library(ONNXRUNTIME_LIB onnxruntime PATHS /usr/local/lib REQUIRED)
find_library(ONNXRUNTIME_CUDA_LIB onnxruntime_providers_cuda PATHS /usr/local/lib)
find_library(ONNXRUNTIME_SHARED_LIB onnxruntime_providers_shared PATHS /usr/local/lib)
include_directories(/usr/local/include)
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")

# Build
add_executable(Light_DT main.cpp ${SRCS})
target_link_libraries(Light_DT
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    ${ONNXRUNTIME_CUDA_LIB}
    ${ONNXRUNTIME_SHARED_LIB}
)
