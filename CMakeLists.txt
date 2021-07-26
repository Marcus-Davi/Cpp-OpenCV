cmake_minimum_required(VERSION 3.10)
project(opencv_test)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

link_libraries(${OpenCV_LIBRARIES})

add_executable(opencv_test src/test.cpp)
add_executable(vo_test src/vo.cpp)
add_executable(vo_test2 src/vo2.cpp)

