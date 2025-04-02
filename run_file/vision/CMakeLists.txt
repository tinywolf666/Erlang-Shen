add_library(vision_lib
        ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yolov12_openvino.cpp
)

target_include_directories(vision_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS})

target_link_libraries(vision_lib ${OpenCV_LIBS})