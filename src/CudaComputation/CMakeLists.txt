project(CudaComputation LANGUAGES CUDA)

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_library(CudaComputation STATIC
    CudaMatrix.h
    CudaMatrix.cu
    CudaMatrix.cuh
    CudaGlobal.cuh
    CudaMath.cuh
    CudaMath.cu)

target_link_libraries(CudaComputation PUBLIC ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})

set_property(TARGET CudaComputation PROPERTY CUDA_ARCHITECTURES 61)
