cmake_minimum_required(VERSION 3.10)

project(MobileNetv2)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)





link_directories(${PROJECT_SOURCE_DIR}/lib/)
include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(/home/verse/C_plus_plus/Graduation_Code/Light_DT/01_Light_DT_code/include/ncnn)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)




find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(MobileNetv2 main.cpp ${SRCS})
target_link_libraries(MobileNetv2 -lopencv_world libncnn.so -lvulkan)
target_link_libraries(MobileNetv2 ${OpenCV_LIBS})





