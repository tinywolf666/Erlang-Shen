add_library(vision_lib
        ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp
)

target_include_directories(vision_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

#   链接OpenCV
target_include_directories(vision_lib PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_lib ${OpenCV_LIBS})