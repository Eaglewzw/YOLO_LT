cmake_minimum_required(VERSION 3.10)

project(MotionDetector)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)


find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


link_directories(${PROJECT_SOURCE_DIR}/lib/)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/home/verse/C_plus_plus/Graduation_Code/Light_DT/01_Light_DT_code/include/ncnn)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/source/*.cpp)


add_executable(MotionDetector main.cpp ${SRCS})


target_link_libraries(MotionDetector -lopencv_world libncnn.so -lvulkan)
target_link_libraries(MotionDetector ${OpenCV_LIBS})





