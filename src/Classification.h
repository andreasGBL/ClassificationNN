#pragma once
#include "CudaMatrix.h"
#include <vector>
#include <string>

typedef float f_t;

void alloc(CudaMatrix<f_t>& All, size_t allocationSize);
void calcAllocationSize(size_t& allocationSize, size_t& tempAllocationVector);
void loadData();
void readIntoVector(std::string filename, std::vector<f_t>& vec);
void forwardPropagation(CudaVector<f_t>& input, CudaVector<f_t>& output);
void backwardPropagation(CudaVector<f_t>& output, CudaVector<f_t>& ExpectedOutput);
void calculateTestsetError(size_t samples);
void train();
void trainMinibatch(std::vector<CudaVector<f_t>>& inputs, std::vector<CudaVector<f_t>>& expectedOutputs);
void classesToOutputs(std::vector<f_t>& classVec, std::vector<f_t>& outputs);
void printAll();