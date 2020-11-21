#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <iostream>
#include <assert.h>
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

#define INSTANTIATE_D2H(Type) template void deviceToHost<Type>(Type*, const Type*, size_t);
#define INSTANTIATE_H2D(Type) template void hostToDevice<Type>(Type*, const Type*, size_t);
INSTANTIATE_D2H(float);
INSTANTIATE_H2D(float);
INSTANTIATE_D2H(double);
INSTANTIATE_H2D(double);
INSTANTIATE_D2H(bool);
INSTANTIATE_H2D(bool);
INSTANTIATE_D2H(char);
INSTANTIATE_H2D(char);
INSTANTIATE_D2H(short);
INSTANTIATE_H2D(short);
INSTANTIATE_D2H(unsigned short);
INSTANTIATE_H2D(unsigned short);
INSTANTIATE_D2H(int);
INSTANTIATE_H2D(int);
INSTANTIATE_D2H(unsigned int);
INSTANTIATE_H2D(unsigned int);
INSTANTIATE_D2H(long long int);
INSTANTIATE_H2D(long long int);
INSTANTIATE_D2H(unsigned long long int);
INSTANTIATE_H2D(unsigned long long int);