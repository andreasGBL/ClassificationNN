#pragma once
#include "CudaMatrix.h"
#include "Classification.cuh"
#include <vector>
#include <string>
#include "MLChart/MLChartFrame.h"

typedef float f_t;
MLChartFrame * chart = nullptr;


void alloc(CudaMatrix<f_t>& All, size_t allocationSize);
void calcAllocationSize(size_t& allocationSize, size_t& tempAllocationVector, size_t& tempAllocationSizeMatrix);
void loadData();
void readIntoVector(std::string filename, std::vector<f_t>& vec);
void forwardPropagation(CudaVector<f_t>& input, CudaVector<f_t>& output);
void backwardPropagation(CudaVector<f_t>& output, CudaVector<f_t>& ExpectedOutput);
std::tuple<double, double> calculateTestsetError(size_t samples);
void train();
void trainMinibatch(std::vector<CudaVector<f_t>>& inputs, std::vector<CudaVector<f_t>>& expectedOutputs, int iteration, bool useCG = false);
void classesToOutputs(std::vector<f_t>& classVec, std::vector<f_t>& outputs);
void printAll();
