#pragma once
#include "CudaGlobal.cuh"

template<typename Real>
void cudaMultElements(Real* a, Real* b, Real* c, size_t n);

template<typename Real>
void cudaFill(Real* a, Real value, size_t n);

template<typename Real>
void cudaRandomize(Real* a, size_t n, unsigned long long seed = 1ULL);

template<typename Real>
void cudaAdd(Real* a, Real* b, Real* c, size_t n);

template<typename Real>
void cudaSub(Real* a, Real* b, Real* c, size_t n);

template<typename Real>
void cudaDivScalar(Real* a, Real* result, Real* scalar, size_t n);

template<typename Real>
void cudaMultScalar(Real* a, Real* result, Real* scalar, size_t n);

template<typename Real>
void cudaAXpY(Real* x, Real* y, Real* result, Real* scalar, size_t n);

template<typename Real>
void cudaSum(Real* a, Real* temp1, Real* result, size_t n);

template<typename Real>
void cudaSqrt(Real* a, Real* result, size_t n);

template<typename Real>
void cudaExp(Real* a, Real* result, size_t n);

template<typename Real>
void cudaTanh(Real* a, Real* result, size_t n);

template<typename Real>
void cudaTanhD(Real* a, Real* result, size_t n);

template<typename Real>
void cudaSigmoid(Real* a, Real* result, size_t n);

template<typename Real>
void cudaSigmoidD(Real* a, Real* result, size_t n);

template<typename Real>
void cudaRELU(Real* a, Real* result, size_t n);

template<typename Real>
void cudaRELUD(Real* a, Real* result, size_t n);





