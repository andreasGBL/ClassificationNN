project(NeuralNetwork LANGUAGES CXX)

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


option(WITH_QT_CHARTS "Provide Charts for the neural network performance (requires QT)." OFF)

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

if(WITH_QT_CHARTS)
add_subdirectory(${SRC_DIR}/MLChart)
include_directories(${SRC_DIR}/MLChart)
endif()

include_directories(${SRC_DIR}/Utilities)
include_directories(${SRC_DIR}/CudaComputation)

find_package(CUDA REQUIRED)

if(Debug)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
endif()

include_directories("${CUDA_TOOLKIT_INCLUDE}")

add_subdirectory(${SRC_DIR}/Regression)
add_subdirectory(${SRC_DIR}/Classification)
add_subdirectory(${SRC_DIR}/CudaComputation)



set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


