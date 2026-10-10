cmake_minimum_required(VERSION 3.14)
project(lepaute_core C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(include)

add_library(lepaute_core SHARED
    src/se3.c
    src/optimization.c
)

target_include_directories(lepaute_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

enable_testing()

add_executable(unit_tests
    tests/test_se3.cpp
)

target_link_libraries(unit_tests PRIVATE
    lepaute_core
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)