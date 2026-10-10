cmake_minimum_required(VERSION 3.16)
project(lepaute_mcu C)

set(CMAKE_C_STANDARD 99)

add_compile_definitions(LEPAUTE_STATIC_MEM)

add_executable(lepaute_mcu
    src/main.c
    src/lepaute.c
    ../core/src/se3.c
    ../core/src/optimization.c
)

target_include_directories(lepaute_mcu PRIVATE
    include
    ../core/include
)

target_compile_options(lepaute_mcu PRIVATE
    -Os
    -Wall
)

if(NOT APPLE)
    target_link_options(lepaute_mcu PRIVATE -Wl,--gc-sections)
    target_compile_options(lepaute_mcu PRIVATE -ffunction-sections -fdata-sections)
endif()

target_link_libraries(lepaute_mcu m)