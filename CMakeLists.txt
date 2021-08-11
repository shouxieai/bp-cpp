cmake_minimum_required(VERSION 2.6)
project(pro)
add_definitions(-std=c++11)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(WORKSPACE ${PROJECT_SOURCE_DIR}/workspace)
set(EXECUTABLE_OUTPUT_PATH ${WORKSPACE})

set(OPENBLAS_DIR ${PROJECT_SOURCE_DIR}/lean/OpenBLAS0.3.17)
include_directories(${OPENBLAS_DIR}/include)
link_directories(${OPENBLAS_DIR}/lib)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wfatal-errors -pthread -w -g -fopenmp")
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(pro ${cpp_srcs})
target_link_libraries(pro openblas)

add_custom_target(
    train
    DEPENDS pro
    WORKING_DIRECTORY ${WORKSPACE}
    COMMAND ./pro train
)

add_custom_target(
    dotest 
    DEPENDS pro
    WORKING_DIRECTORY ${WORKSPACE}
    COMMAND ./pro test
)