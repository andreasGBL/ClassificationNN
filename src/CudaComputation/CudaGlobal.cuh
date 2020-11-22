#pragma once
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <iostream>
#include <assert.h>
#include <Templates.h>

#define TPB 1024
#define DIVRND(a,b) ((a+b-1)/b)
#define CUDACALL(ans) checkCudaError((ans), __FILE__, __LINE__)
inline bool checkCudaError(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		std::cout << "Error on CUDA call: "<< cudaGetErrorString(code)<<" "<< " " << file<<" " <<line << std::endl;
		return false;
	}
	return true;
}
#define CUBLASCALL(ans) checkCublasError((ans), __FILE__, __LINE__)
inline bool checkCublasError(cublasStatus_t code, const char* file, int line)
{
	if (code != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "Error on CUBLAS call: " << code << " " << " " << file << " " << line << std::endl;
		return false;
	}
	return true;
}

template<typename T>
inline void deviceToHost(T* destHost, const T* srcDev, size_t n) { CUDACALL(cudaMemcpy((void*)destHost, (void*)srcDev, sizeof(T) * n, cudaMemcpyDeviceToHost)); }
template<typename T>
inline void hostToDevice(T* destDev, const T* srcHost, size_t n) { CUDACALL(cudaMemcpy((void*)destDev, (void*)srcHost, sizeof(T) * n, cudaMemcpyHostToDevice)); }
template<typename T>
inline void deviceToDevice(T * destDev, const T * srcDev, size_t n) { CUDACALL(cudaMemcpy((void *)destDev, (void *)srcDev, sizeof(T) * n, cudaMemcpyDeviceToDevice)); }

#define INSTANTIATE_D2H(Type) template void deviceToHost<Type>(Type*, const Type*, size_t);
#define INSTANTIATE_H2D(Type) template void hostToDevice<Type>(Type*, const Type*, size_t);
#define INSTANTIATE_D2D(Type) template void deviceToDevice<Type>(Type*, const Type*, size_t);
#define INSTANTIATE_MEMCPYS(Type) INSTANTIATE_D2H(Type) INSTANTIATE_H2D(Type) INSTANTIATE_D2D(Type) 
EXECUTE_MACRO_FOR_ALL_TYPES(INSTANTIATE_MEMCPYS);