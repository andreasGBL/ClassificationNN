project(ClassificationNN LANGUAGES CXX)

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# find_package(CUDA REQUIRED)
# if(Debug)
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61 -lineinfo")
# else()
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
# endif()

# option(WITH_QT_CHARTS "Provide Charts for the neural network performance (requires QT)." OFF)
# include_directories("${CUDA_TOOLKIT_INCLUDE}")
#set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")


cuda_add_executable(ClassificationNN
	Classification.cpp
	Classification.h
)

target_link_libraries(ClassificationNN ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
target_link_libraries(ClassificationNN CudaComputation)
if(WITH_QT_CHARTS)
target_compile_definitions(ClassificationNN PRIVATE WITH_QT_CHARTS)
target_link_libraries(ClassificationNN MLChart)
endif()
set_property(TARGET ClassificationNN PROPERTY CUDA_ARCHITECTURES 61)


set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(DATA_DIR ${CMAKE_SOURCE_DIR}/data)
configure_file(${DATA_DIR}/mnist_train_in_50000.txt ./data/mnist_train_in_50000.txt COPYONLY)
configure_file(${DATA_DIR}/mnist_train_out_50000.txt ./data/mnist_train_out_50000.txt COPYONLY)
configure_file(${DATA_DIR}/mnist_test_in_10000.txt ./data/mnist_test_in_10000.txt COPYONLY)
configure_file(${DATA_DIR}/mnist_test_out_10000.txt ./data/mnist_test_out_10000.txt COPYONLY)