#include <cub/device/device_reduce.cuh>
#include "CudaMatrix.cuh"
#include <curand.h>
#include <curand_kernel.h>


template<typename Real>
__device__ Real sigmoid(Real x) {
	Real e = exp(-x);
	return Real(1) / (Real(1) + e);
}

template<typename Real>
__device__ Real RELU(Real x) {
	return x > Real(0) ? x : Real(0);
}

template<typename Real>
__device__ Real RELUD(Real x) {
	return x > Real(0) ? Real(1) : Real(0);
}


template<typename Real>
__global__ void multElementsKernel(Real* a, Real* b, Real* c, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		c[i] = a[i] * b[i];
	}
}

template<typename Real>
__global__ void sqrtKernel(Real* a, Real* result, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		result[i] = sqrt(a[i]);
	}
}

template<typename Real>
__global__ void fillKernel(Real* a, Real value, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		a[i] = value;
	}
}
template<typename Real>
__global__ void addKernel(Real* a, Real* b, Real* c, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

template<typename Real>
__global__ void subKernel(Real* a, Real* b, Real* c, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		c[i] = a[i] - b[i];
	}
}

template<typename Real>
__global__ void randomizeKernel(Real* a, unsigned long long seed, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		curandStateXORWOW_t state;
		curand_init(seed, i, 0, &state);
		a[i] = (Real)curand_uniform(&state)-(Real)0.5;
	}
}

template<typename Real>
__global__ void multScalarKernel(Real* a, Real* result, Real* scalar, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		result[i] = *scalar * a[i];
	}
}

template<typename Real>
__global__ void axpyKernel(Real* x, Real* y, Real* result, Real* scalar, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		result[i] = *scalar * x[i] + y[i];
	}
}

template<typename Real>
__global__ void divScalarKernel(Real* a, Real* result, Real* scalar, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		result[i] = a[i] / *scalar;
	}
}

template <typename Real>
__global__ void expKernel(Real* a, Real* result, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		result[i] = exp(a[i]);
	}
}

template <typename Real>
__global__ void tanhKernel(Real* a, Real* result, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		result[i] = tanh(a[i]);
	}
}

template <typename Real>
__global__ void tanhDKernel(Real* a, Real* result, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		Real t = tanh(a[i]);
		result[i] = (Real) 1.0 - (t * t);
	}
}

template <typename Real>
__global__ void sigmoidKernel(Real* a, Real* result, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		result[i] = sigmoid(a[i]);
	}
}

template <typename Real>
__global__ void sigmoidDKernel(Real* a, Real* result, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		Real s = sigmoid(a[i]);
		result[i] = s * (Real(1) - s);
	}
}

template <typename Real>
__global__ void RELUKernel(Real* a, Real* result, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		result[i] = RELU(a[i]);
	}
}

template <typename Real>
__global__ void RELUDKernel(Real* a, Real* result, size_t n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		result[i] = RELUD(a[i]);
	}
}

template<typename Real>
void cudaMultElements(Real* a, Real* b, Real* c, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	multElementsKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, b, c, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaMultScalar(Real* a, Real* result, Real* scalar, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	multScalarKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, result, scalar, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaAXpY(Real* x, Real* y, Real* result, Real* scalar, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	axpyKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (x, y, result, scalar, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaDivScalar(Real* a, Real* result, Real* scalar, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	divScalarKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, result, scalar, n);
	assert(CUDACALL(cudaGetLastError()));
}


template<typename Real>
void cudaFill(Real* a, Real value, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	fillKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, value, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaRandomize(Real* a, size_t n, unsigned long long seed) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	randomizeKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, seed, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaAdd(Real* a, Real* b, Real* c, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	addKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, b, c, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaSub(Real* a, Real* b, Real* c, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	subKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, b, c, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaSum(Real* a, Real* temp1, Real* result, size_t n) {
	void* temp_storage = reinterpret_cast<void*>(temp1);
	size_t temp_storage_size = 0;
	size_t available_size = n * sizeof(Real);
	CUDACALL(cub::DeviceReduce::Sum(nullptr, temp_storage_size, a, result, (int)n));
	if (available_size + 1 < temp_storage_size) {
		void* storage;
		CUDACALL(cudaMalloc(&storage, temp_storage_size));
		CUDACALL(cub::DeviceReduce::Sum(storage, temp_storage_size, a, result, (int)n));
		CUDACALL(cudaFree(storage));
	}
	else {
		CUDACALL(cub::DeviceReduce::Sum(temp_storage, temp_storage_size, a, result, (int)n));
	}
	assert(CUDACALL(cudaGetLastError()));
}


template<typename Real>
void cudaExp(Real* a, Real* result, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	expKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, result, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaTanh(Real* a, Real* result, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	tanhKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, result, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaTanhD(Real* a, Real* result, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	tanhDKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, result, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaSigmoid(Real* a, Real* result, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	sigmoidKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, result, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaSigmoidD(Real* a, Real* result, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	sigmoidDKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, result, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaRELU(Real* a, Real* result, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	RELUKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, result, n);
	assert(CUDACALL(cudaGetLastError()));
}

template<typename Real>
void cudaRELUD(Real* a, Real* result, size_t n) {
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	RELUDKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, result, n);
	assert(CUDACALL(cudaGetLastError()));
}


template<typename Real>
void cudaSqrt(Real* a, Real* result, size_t n)
{
	unsigned int BLOCKSIZE = n < TPB ? (unsigned int)n : TPB;
	unsigned int GRIDSIZE = DIVRND(n, BLOCKSIZE);
	sqrtKernel<Real> << <GRIDSIZE, BLOCKSIZE >> > (a, result, n);
	assert(CUDACALL(cudaGetLastError()));
}


#define INSTANTIATE_EXP(Real) template void cudaExp<Real>(Real*, Real*, size_t);
INSTANTIATE_EXP(float);
INSTANTIATE_EXP(double);
INSTANTIATE_EXP(half);

#define INSTANTIATE_TANH(Real) template void cudaTanh<Real>(Real*, Real*, size_t);
INSTANTIATE_TANH(float);
INSTANTIATE_TANH(double);
INSTANTIATE_TANH(half);

#define INSTANTIATE_TANHD(Real) template void cudaTanhD<Real>(Real*, Real*, size_t);
INSTANTIATE_TANHD(float);
INSTANTIATE_TANHD(double);
INSTANTIATE_TANHD(half);

#define INSTANTIATE_SIGMOID(Real) template void cudaSigmoid<Real>(Real*, Real*, size_t);
INSTANTIATE_SIGMOID(float);
INSTANTIATE_SIGMOID(double);
INSTANTIATE_SIGMOID(half);

#define INSTANTIATE_SIGMOIDD(Real) template void cudaSigmoidD<Real>(Real*, Real*, size_t);
INSTANTIATE_SIGMOIDD(float);
INSTANTIATE_SIGMOIDD(double);
INSTANTIATE_SIGMOIDD(half);

#define INSTANTIATE_RELU(Real) template void cudaRELU<Real>(Real*, Real*, size_t);
INSTANTIATE_RELU(float);
INSTANTIATE_RELU(double);
INSTANTIATE_RELU(half);

#define INSTANTIATE_RELUD(Real) template void cudaRELUD<Real>(Real*, Real*, size_t);
INSTANTIATE_RELUD(float);
INSTANTIATE_RELUD(double);
INSTANTIATE_RELUD(half);

#define INSTANTIATE_SQRT(Real) template void cudaSqrt<Real>(Real*, Real*, size_t);
INSTANTIATE_SQRT(float);
INSTANTIATE_SQRT(double);
INSTANTIATE_SQRT(half);

#define INSTANTIATE_SUM(Real) template void cudaSum<Real>(Real*, Real*, Real*, size_t);
INSTANTIATE_SUM(float);
INSTANTIATE_SUM(double);
INSTANTIATE_SUM(half);

#define INSTANTIATE_SUB(Real) template void cudaSub<Real>(Real*, Real*, Real*, size_t);
INSTANTIATE_SUB(float);
INSTANTIATE_SUB(double);
INSTANTIATE_SUB(half);

#define INSTANTIATE_ADD(Real) template void cudaAdd<Real>(Real*, Real*, Real*, size_t);
INSTANTIATE_ADD(float);
INSTANTIATE_ADD(double);
INSTANTIATE_ADD(half);

#define INSTANTIATE_MULTELEMENTS(Real) template void cudaMultElements<Real>(Real*, Real*, Real*, size_t);
INSTANTIATE_MULTELEMENTS(float);
INSTANTIATE_MULTELEMENTS(double);
INSTANTIATE_MULTELEMENTS(half);

#define INSTANTIATE_RAND(Real) template void cudaRandomize<Real>(Real*, size_t, unsigned long long);
INSTANTIATE_RAND(float);
INSTANTIATE_RAND(double);
INSTANTIATE_RAND(half);

#define INSTANTIATE_FILL(Real) template void cudaFill<Real>(Real*, Real, size_t);
INSTANTIATE_FILL(float);
INSTANTIATE_FILL(double);
INSTANTIATE_FILL(half);

#define INSTANTIATE_MULTSCALAR(Real) template void cudaMultScalar<Real>(Real*, Real*, Real*, size_t);
INSTANTIATE_MULTSCALAR(float);
INSTANTIATE_MULTSCALAR(double);
INSTANTIATE_MULTSCALAR(half);

#define INSTANTIATE_AXPY(Real) template void cudaAXpY<Real>(Real*, Real*, Real*, Real*, size_t);
INSTANTIATE_AXPY(float);
INSTANTIATE_AXPY(double);
INSTANTIATE_AXPY(half);

#define INSTANTIATE_DIVSCALAR(Real) template void cudaDivScalar<Real>(Real*, Real*, Real*, size_t);
INSTANTIATE_DIVSCALAR(float);
INSTANTIATE_DIVSCALAR(double);
INSTANTIATE_DIVSCALAR(half);

