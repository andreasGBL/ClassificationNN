#pragma once
#include "CudaMatrix.cuh"
#include <string>
#include <Templates.h>

template<typename Real>
class CudaVector;

template<typename Real>
class CudaMatrixRepr {
public:
	CudaMatrixRepr();
	CudaMatrixRepr(size_t rows, size_t cols);
	CudaMatrixRepr(Real * data, size_t rows, size_t cols);
	~CudaMatrixRepr();

	void init(Real * data, size_t rows, size_t cols);
	void init(size_t rows, size_t cols);

	CudaVector<Real> getColumn(size_t col) { return CudaVector<Real>(data + (rows * col), rows); }

	void print();

	Real * getData() { return data; }

	size_t numCols() { return cols; }
	size_t numRows() { return rows; }
	size_t numElems() { return rows * cols; }

	static void freeCublasHandle();
	static cublasHandle_t handle;
protected:
	void initHandle();
	Real * data;
	size_t rows = 0, cols = 0;
	bool ownership = true;
	bool initialized = false;

	bool checkDims(CudaMatrixRepr<Real> & B);
	bool checkDims(CudaMatrixRepr<Real> & B, CudaMatrixRepr<Real> & C);
	std::string toString(Real r);
};

/**
Col_Major Cuda Matrix (Wrapper Class for cuBLAS)
*/
template<typename Real>
class CudaMatrix : public CudaMatrixRepr<Real>
{
public:
	CudaMatrix();
	CudaMatrix(size_t rows, size_t cols);
	CudaMatrix(Real * data, size_t rows, size_t cols);
	
	void multElements(CudaMatrix<Real>& B, CudaMatrix<Real>& Result);
	void add(CudaMatrix<Real>& B, CudaMatrix<Real>& Result);
	void sub(CudaMatrix<Real>& B, CudaMatrix<Real>& Result);
	void sum(CudaMatrix<Real>& temp, Real* result);
	void dot(CudaMatrix<Real>& B, CudaMatrix<Real>& temp1, CudaMatrix<Real>& temp2, Real* result);
	void norm2(CudaMatrix<Real>& temp1, CudaMatrix<Real>& temp2, Real* result);
	void multScalar(CudaMatrix<Real>& Result, Real* scalar);
	void divScalar(CudaMatrix<Real>& Result, Real* scalar);
	void fill(Real value);
	void randomize(unsigned long long seed = 1ULL);
	void mult(CudaMatrix<Real>& B, CudaMatrix<Real>& Result, bool transposeA = false, bool transposeB = false);
	void exp(CudaMatrix<Real>& Result);
	void softMax(CudaMatrix<Real>& Result, CudaMatrix<Real>& temp);
	void tanh(CudaMatrix<Real>& Result);
	void tanhD(CudaMatrix<Real>& Result);
	void sigmoid(CudaMatrix<Real>& Result);
	void sigmoidD(CudaMatrix<Real>& Result);
	void RELU(CudaMatrix<Real>& Result);
	void RELUD(CudaMatrix<Real>& Result);
	void axpy(CudaMatrix<Real>& Y, CudaMatrix<Real>& Result, Real* Scalar);
	CudaMatrix<Real> getSubmatrix(size_t firstCol, size_t lastCol);

	void copyFrom(CudaMatrix<Real>& SRC);

protected:
	bool checkDimsMult(CudaMatrix<Real>& B, CudaMatrix<Real>& C, bool transA = false, bool transB = false);
};
/**
* Cuda Vector Class (Cuda Matrix with one Column)
*/
template<typename Real>
class CudaVector : public CudaMatrix<Real> {
public:
	CudaVector(size_t n);
	CudaVector(Real* data, size_t n);
	CudaVector();
	CudaMatrix<Real> getSubmatrix(size_t firstCol, size_t lastCol) = delete;
	CudaVector<Real> getSubvector(size_t firstElem, size_t lastElem);
	void initVector(Real* data, size_t n);
	void initVector(size_t n);
	void init(Real * data, size_t rows, size_t cols) = delete;
	void init(size_t rows, size_t cols) = delete;
	CudaVector<Real> getColumn(size_t col) = delete;
};

template<typename Real>
CudaVector<Real>::CudaVector(size_t n) :
	CudaMatrix<Real>(n, 1) {

}

template<typename Real>
CudaVector<Real>::CudaVector(Real* data, size_t n) :
	CudaMatrix<Real>(data, n, 1)
{

}

template<typename Real>
CudaVector<Real>::CudaVector() :
	CudaMatrix<Real>()
{
}

template<typename Real>
CudaVector<Real> CudaVector<Real>::getSubvector(size_t firstElem, size_t lastElem)
{
	if(lastElem >= firstElem)
		return CudaVector<Real>(this->getData()+ firstElem, lastElem - firstElem + 1);
	else
		return CudaVector<Real>(this->getData() + firstElem, 0);
}

template<typename Real>
void CudaVector<Real>::initVector(Real* dataptr, size_t n)
{
	CudaMatrix<Real>::init(dataptr, n, 1);
}

template<typename Real>
void CudaVector<Real>::initVector(size_t n)
{
	CudaMatrix<Real>::init(n, 1);
}

template<typename Real>
cublasHandle_t CudaMatrixRepr<Real>::handle = nullptr;

template<typename Real>
CudaMatrixRepr<Real>::CudaMatrixRepr() :
	CudaMatrixRepr(0,0)
{
}



template<typename Real>
CudaMatrixRepr<Real>::CudaMatrixRepr(size_t rows, size_t cols)
{
	init(rows, cols);
}



template<typename Real>
CudaMatrixRepr<Real>::~CudaMatrixRepr()
{
	if (ownership && initialized) {
		CUDACALL(cudaFree(data));
	}
}

template<typename Real>
void CudaMatrixRepr<Real>::init(Real* data, size_t rows, size_t cols)
{
	this->cols = cols;
	this->rows = rows;
	this->data = data;
	ownership = false;
	initialized = true;
	initHandle();
}

template<typename Real>
void CudaMatrixRepr<Real>::init(size_t newRows, size_t newCols)
{
	cols = newCols;
	rows = newRows;
	ownership = true;
	initialized = false;
	if (newCols * newRows != 0) {
		CUDACALL(cudaMalloc((void**)&data, sizeof(Real) * numElems()));
		initialized = true;
	}
	initHandle();
}

template<typename Real>
std::string CudaMatrixRepr<Real>::toString(Real r)
{
	return std::to_string(r);
}

template<>
std::string CudaMatrixRepr<float2>::toString(float2 r)
{
	return std::string("[" + std::to_string(r.x) + ", " + std::to_string(r.y) + "]");
}

template<>
std::string CudaMatrixRepr<double2>::toString(double2 r)
{
	return std::string("[" + std::to_string(r.x) + ", " + std::to_string(r.y) + "]");
}

template<>
std::string CudaMatrixRepr<double3>::toString(double3 r)
{
	return std::string("[" + std::to_string(r.x) + ", " + std::to_string(r.y) + ", " + std::to_string(r.z) + "]");
}

template<>
std::string CudaMatrixRepr<float3>::toString(float3 r)
{
	return std::string("[" + std::to_string(r.x) + ", " + std::to_string(r.y) + ", " + std::to_string(r.z) + "]");
}

template<>
std::string CudaMatrixRepr<double4>::toString(double4 r)
{
	return std::string("[" + std::to_string(r.x) + ", " + std::to_string(r.y) + ", " + std::to_string(r.z) + ", " + std::to_string(r.w) + "]");
}

template<>
std::string CudaMatrixRepr<float4>::toString(float4 r)
{
	return std::string("[" + std::to_string(r.x) + ", " + std::to_string(r.y) + ", " + std::to_string(r.z) + ", " + std::to_string(r.w) + "]");
}


template<typename Real>
CudaMatrixRepr<Real>::CudaMatrixRepr(Real * data, size_t rows, size_t cols)
{
	init(data, rows, cols);
}

template<typename Real>
CudaMatrix<Real>::CudaMatrix() :
	CudaMatrixRepr()
{
}

template<typename Real>
CudaMatrix<Real>::CudaMatrix(size_t rows, size_t cols) :
	CudaMatrixRepr(rows, cols)
{

}
template<typename Real>
CudaMatrix<Real>::CudaMatrix(Real * data, size_t rows, size_t cols) :
	CudaMatrixRepr(data, rows, cols)
{
}

template<typename Real>
void CudaMatrix<Real>::multElements(CudaMatrix<Real>& B, CudaMatrix<Real>& Result)
{
	bool ok = checkDims(B, Result);
	assert(ok);
	if (ok)
		cudaMultElements<Real>(data, B.data, Result.data, (size_t) numElems());
}

template<typename Real>
void CudaMatrix<Real>::add(CudaMatrix<Real>& B, CudaMatrix<Real>& Result)
{
	bool ok = checkDims(B, Result);
	assert(ok);
	if (ok)
		cudaAdd<Real>(data, B.data, Result.data, (size_t)numElems());
}

template<typename Real>
void CudaMatrix<Real>::sub(CudaMatrix<Real>& B, CudaMatrix<Real>& Result)
{
	bool ok = checkDims(B, Result);
	assert(ok);
	if (ok)
		cudaSub<Real>(data, B.data, Result.data, (size_t)numElems());
}

template<typename Real>
void CudaMatrix<Real>::sum(CudaMatrix<Real>& temp, Real* result)
{
	bool ok = checkDims(temp);
	assert(ok);
	if (ok)
		cudaSum<Real>(data, temp.data, result, (size_t)numElems());
}

template<typename Real>
void CudaMatrix<Real>::dot(CudaMatrix<Real>& B, CudaMatrix<Real>& temp1, CudaMatrix<Real>& temp2, Real* result)
{
	bool ok = checkDims(B, temp1) && checkDims(temp2);
	assert(ok);
	if (ok) {
		size_t n = (size_t) numElems();
		cudaMultElements<Real>(data, B.data, temp1.data, n);
		cudaSum<Real>(temp1.data, temp2.data, result, n);
	}
}

template<typename Real>
void CudaMatrix<Real>::norm2(CudaMatrix<Real>& temp1, CudaMatrix<Real>& temp2, Real* result)
{
	dot(*this, temp1, temp2, result);
	cudaSqrt<Real>(result, result, 1);
}

template<typename Real>
void CudaMatrix<Real>::multScalar(CudaMatrix<Real>& Result, Real* scalar)
{
	bool ok = checkDims(Result);
	assert(ok);
	if (ok)
		cudaMultScalar<Real>(data, Result.data, scalar, (size_t)numElems());
}

template<typename Real>
void CudaMatrix<Real>::divScalar(CudaMatrix<Real>& Result, Real* scalar)
{
	bool ok = checkDims(Result);
	assert(ok);
	if (ok)
		cudaDivScalar<Real>(data, Result.data, scalar, (size_t)numElems());
}

template<typename Real>
void CudaMatrix<Real>::fill(Real value)
{
	cudaFill<Real>(data, value, (size_t)numElems());
}

template<typename Real>
void CudaMatrix<Real>::randomize(unsigned long long seed)
{
	cudaRandomize<Real>(data, (size_t)numElems(), seed);
}
template<typename Real>
struct multImpl {
	static void apply(CudaMatrix<Real> & A, CudaMatrix<Real> & B, CudaMatrix<Real> & Result, bool transposeA, bool transposeB)
	{
		assert(false);
	}
};

template<>
struct multImpl<float> {
	static void apply(CudaMatrix<float>& A, CudaMatrix<float>& B, CudaMatrix<float>& Result, bool transposeA, bool transposeB) {
		float alpha = 1.0f;
		float beta = 0.0f;
		cublasOperation_t ta = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t tb = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
		//mxn = mxk * kxn
		int m = (int)Result.numRows();
		int n = (int)Result.numCols();
		int k = transposeA ? (int)A.numRows() : (int) A.numCols();
		CUBLASCALL(cublasSgemm(CudaMatrixRepr<float>::handle, ta, tb, m, n, k, &alpha, A.getData(), A.numRows(), B.getData(), B.numRows(), &beta, Result.getData(), Result.numRows()));
	}
};
template<>
struct multImpl<double> {
	static void apply(CudaMatrix<double>& A, CudaMatrix<double>& B, CudaMatrix<double>& Result, bool transposeA, bool transposeB) {
		double alpha = 1.0;
		double beta = 0.0;
		cublasOperation_t ta = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t tb = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
		//mxn = mxk * kxn
		int m = (int)Result.numRows();
		int n = (int)Result.numCols();
		int k = transposeA ? (int)A.numRows() : (int)A.numCols();
		CUBLASCALL(cublasDgemm(CudaMatrixRepr<double>::handle, ta, tb, m, n, k, &alpha, A.getData(), A.numRows(), B.getData(), B.numRows(), &beta, Result.getData(), Result.numRows()));
	}
};


template<typename Real>
void CudaMatrix<Real>::mult(CudaMatrix<Real>& B, CudaMatrix<Real>& Result, bool transposeA, bool transposeB)
{
	bool ok = checkDimsMult(B, Result, transposeA, transposeB);
	assert(ok);
	if (ok) {
		multImpl<Real>::apply(*this, B, Result, transposeA, transposeB);
	}

}

template<typename Real>
void CudaMatrix<Real>::exp(CudaMatrix<Real>& Result)
{
	bool ok = checkDims(Result);
	assert(ok);
	if (ok)
		cudaExp<Real>(data, Result.data, numElems());
}

template<typename Real>
void CudaMatrix<Real>::softMax(CudaMatrix<Real>& Result, CudaMatrix<Real>& temp)
{
	bool ok = checkDims(Result, temp);
	assert(ok);
	if (ok) {
		cudaExp<Real>(data, Result.data, numElems());
		cudaSum<Real>(Result.data, temp.data+1, temp.data, numElems());
		cudaDivScalar<Real>(Result.data, Result.data, temp.data, numElems());
	}
}

template<typename Real>
void CudaMatrix<Real>::tanh(CudaMatrix<Real>& Result)
{
	bool ok = checkDims(Result);
	assert(ok);
	if (ok)
		cudaTanh<Real>(data, Result.data, numElems());
}

template<typename Real>
void CudaMatrix<Real>::tanhD(CudaMatrix<Real>& Result)
{
	bool ok = checkDims(Result);
	assert(ok);
	if (ok)
		cudaTanhD<Real>(data, Result.data, numElems());
}

template<typename Real>
inline void CudaMatrix<Real>::sigmoid(CudaMatrix<Real>& Result)
{
	bool ok = checkDims(Result);
	assert(ok);
	if (ok)
		cudaSigmoid<Real>(data, Result.data, numElems());
}

template<typename Real>
inline void CudaMatrix<Real>::sigmoidD(CudaMatrix<Real>& Result)
{
	bool ok = checkDims(Result);
	assert(ok);
	if (ok)
		cudaSigmoidD<Real>(data, Result.data, numElems());
}

template<typename Real>
inline void CudaMatrix<Real>::RELU(CudaMatrix<Real>& Result)
{
	bool ok = checkDims(Result);
	assert(ok);
	if (ok)
		cudaRELU<Real>(data, Result.data, numElems());
}

template<typename Real>
inline void CudaMatrix<Real>::RELUD(CudaMatrix<Real>& Result)
{
	bool ok = checkDims(Result);
	assert(ok);
	if (ok)
		cudaRELUD<Real>(data, Result.data, numElems());
}

template<typename Real>
void CudaMatrix<Real>::axpy(CudaMatrix<Real>& Y, CudaMatrix<Real>& Result, Real* Scalar)
{
	bool ok = checkDims(Y, Result);
	assert(ok);
	if (ok)
		cudaAXpY<Real>(data, Y.data, Result.data, Scalar, numElems());
}

template<typename Real>
void CudaMatrixRepr<Real>::print()
{
	std::vector<Real> host(numElems());
	deviceToHost<Real>(&host[0], data, numElems());
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			std::cout << toString(host[row + col * rows]).c_str()<<" ";
		}
		std::cout << std::endl;
	}
	std::cout << "--------------------------------" << std::endl;
}

template<typename Real>
CudaMatrix<Real> CudaMatrix<Real>::getSubmatrix(size_t firstCol, size_t lastCol)
{
	if (lastCol >= firstCol)
		return CudaMatrix<Real>(getColumn(firstCol).getData(), rows, lastCol - firstCol + 1);
	else
		return CudaMatrix<Real>(getColumn(firstCol).getData(), 0, 0);
}

template<typename Real>
void CudaMatrix<Real>::copyFrom(CudaMatrix<Real>& SRC)
{
	bool ok = checkDims(SRC);
	assert(ok);
	if(ok)
		CUDACALL(cudaMemcpy(data, SRC.data, sizeof(Real) * (size_t)numElems(), cudaMemcpyDeviceToDevice));
}

template<typename Real>
void CudaMatrixRepr<Real>::freeCublasHandle()
{
	if (CudaMatrixRepr<Real>::handle) {
		cublasDestroy_v2(CudaMatrixRepr<Real>::handle);
		CudaMatrixRepr<Real>::handle = nullptr;
	}

}

template<typename Real>
void CudaMatrixRepr<Real>::initHandle() {
	if (!CudaMatrixRepr<Real>::handle)
		cublasCreate_v2(&CudaMatrixRepr<Real>::handle);
}

template<typename Real>
bool CudaMatrixRepr<Real>::checkDims(CudaMatrixRepr<Real>& B)
{
	return rows <= B.rows && cols <= B.cols;
}

template<typename Real>
bool CudaMatrixRepr<Real>::checkDims(CudaMatrixRepr<Real>& B, CudaMatrixRepr<Real>& C)
{
	return checkDims(B) && checkDims(C);
}

template<typename Real>
bool CudaMatrix<Real>::checkDimsMult(CudaMatrix<Real>& B, CudaMatrix<Real>& C, bool transA, bool transB)
{
	//C = A*B
	//(lxn) = (lxm)(mxn)
	size_t l = C.rows, n = C.cols;
	bool ok = true;
	ok &= l == (transA ? cols : rows); //check l
	ok &= (transA ? rows : cols) == (transB ? B.cols : B.rows); //check m
	ok &= n == (transB ? B.rows : B.cols);
	return ok;
}

#define INSTANTIATE_CUDAMATRIX_REPR(Real) template class CudaMatrixRepr<Real>;
#define INSTANTIATE_CUDAMATRIX(Real) template class CudaMatrix<Real>;
#define INSTANTIATE_CUDAVECTOR(Real) template class CudaVector<Real>;
EXECUTE_MACRO_FOR_REAL_TYPES(INSTANTIATE_CUDAMATRIX);
EXECUTE_MACRO_FOR_REAL_TYPES(INSTANTIATE_CUDAVECTOR);
EXECUTE_MACRO_FOR_ADVANCED_REAL_TYPES(INSTANTIATE_CUDAMATRIX_REPR);