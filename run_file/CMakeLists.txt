cmake_minimum_required(VERSION 3.10)
project(erlangShen)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
add_subdirectory(vision)

# 创建 main 可执行程序
add_executable(erlangShen core/src/core.cpp)
target_link_libraries(erlangShen PRIVATE vision_lib ${OpenCV_LIBS})

# 创建 test 可执行程序
#add_executable(my_test test.cpp)
#target_link_libraries(my_test PRIVATE myfuncs)

