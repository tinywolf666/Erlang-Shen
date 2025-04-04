add_library(config
        ${CMAKE_CURRENT_SOURCE_DIR}/src/config.cpp
)

target_include_directories(config PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(config yaml-cpp)