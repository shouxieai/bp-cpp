cmake_minimum_required(VERSION 2.6)
project(pro)
add_definitions(-std=c++11)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

# 因为protobuf，需要用特定版本，所以这里指定路径
set(OPENBLAS_DIR ${PROJECT_SOURCE_DIR}/lean/OpenBLAS0.3.17)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/tensorRT
    ${PROJECT_SOURCE_DIR}/src/tensorRT/common
    ${OpenCV_INCLUDE_DIRS}
    ${OPENBLAS_DIR}/include
)

# 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# 这可能带来错误
link_directories(
    ${OPENBLAS_DIR}/lib
)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wfatal-errors -pthread -w -g -fopenmp")
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(pro ${cpp_srcs})
target_link_libraries(pro openblas)

add_custom_target(
    train
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro train
)

add_custom_target(
    dotest 
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro test
)