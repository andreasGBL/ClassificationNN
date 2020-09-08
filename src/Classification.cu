#include "Classification.cuh"
#include "CudaGlobal.cuh"






template<typename Real>
__global__ void divideKernel(Real* a, Real* b, Real* result) {
	*result = *a / *b;
}



template<typename Real>
__global__ void multKernel(Real* a, Real* b, Real* result) {
	*result = *a * *b;
}

template<typename Real>
void divide(Real* a, Real* b, Real* result) {
	divideKernel<Real> << <1, 1 >> > (a, b, result);
}

template void divide<float>(float*, float*, float*);
template void divide<double>(double*, double*, double*);

template<typename Real>
void mult(Real* a, Real* b, Real* result) {
	multKernel<Real> << <1, 1 >> > (a, b, result);
}

template void mult<float>(float*, float*, float*);
template void mult<double>(double*, double*, double*);