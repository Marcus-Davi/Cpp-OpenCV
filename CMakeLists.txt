cmake_minimum_required(VERSION 3.10)
project(opencv_test)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_test src/test.cpp)
target_link_libraries(opencv_test ${OpenCV_LIBS})
