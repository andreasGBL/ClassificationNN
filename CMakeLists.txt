project(ClassificationNN)
cmake_minimum_required(VERSION 3.14)
find_package(CUDA REQUIRED)


if(Debug)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61 -lineinfo")
else()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
endif()

include_directories(${CMAKE_SOURCE_DIR}/inc)
set(SRC_DIR "./src")

cuda_add_executable(ClassificationNN
	${SRC_DIR}/Classification.cpp
	${SRC_DIR}/Classification.h
	${SRC_DIR}/CudaMatrix.h
	${SRC_DIR}/CudaMatrix.cuh
	${SRC_DIR}/CudaMatrix.cu
	${SRC_DIR}/CudaGlobal.cuh
	${SRC_DIR}/Timer.h
)
target_link_libraries(ClassificationNN
            ${CUDA_cublas_LIBRARY}
            ${CUDA_curand_LIBRARY})

set_property(TARGET ClassificationNN PROPERTY CXX_STANDARD 20)
set_property(TARGET ClassificationNN PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ClassificationNN PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
