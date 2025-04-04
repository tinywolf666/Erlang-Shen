cmake_minimum_required(VERSION 3.20)
project(erlangShen)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(yaml-cpp REQUIRED)

add_subdirectory(vision)
add_subdirectory(config)


# 创建 main 可执行程序
add_executable(erlangShen core/src/core.cpp)

target_link_libraries(erlangShen
        PRIVATE
        vision_lib
        ${OpenCV_LIBS}
        openvino::runtime
        config)

target_include_directories(erlangShen PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)
# 创建 test 可执行程序
#add_executable(my_test test.cpp)
#target_link_libraries(my_test PRIVATE myfuncs)

