#include "CudaMath.cuh"
#include "CudaGlobal.cuh"
#include <Templates.h>






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
#define INSTANTIATE_DIVIDE(Real) template void divide<Real>(Real*, Real*, Real*);
EXECUTE_MACRO_FOR_REAL_TYPES(INSTANTIATE_DIVIDE);

template<typename Real>
void mult(Real* a, Real* b, Real* result) {
	multKernel<Real> << <1, 1 >> > (a, b, result);
}

#define INSTANTIATE_MULT(Real) template void mult<Real>(Real*, Real*, Real*);
EXECUTE_MACRO_FOR_REAL_TYPES(INSTANTIATE_MULT);