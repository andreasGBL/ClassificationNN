#include "Classification.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include "Timer.h"
#include <random>

const size_t nodesPerLayer[] = { 784, 40, 20, 10 };

const size_t layers = sizeof(nodesPerLayer) / sizeof(nodesPerLayer[0]);
const size_t inputSize = nodesPerLayer[0], outputSize = nodesPerLayer[layers - 1];
const size_t testDataSize = 10000, trainDataSize = 60000;
const size_t minibatchSize = 100;
const size_t minibatchIterations = 100000;
const size_t classes = 10;
const f_t negative_learn_rate = -0.05f;
const f_t constants[] = { (f_t)minibatchSize, negative_learn_rate, (f_t)minibatchIterations};
std::mt19937 gen(1);
std::uniform_int_distribution<int> distrib(0, (int)trainDataSize - 1);

const unsigned long long seed = 42;

std::vector<CudaMatrix<f_t>> weights, dw, mbw;
std::vector<CudaVector<f_t>> biases, db, mbb, a, z, trainIn, trainOut, testIn, testOut;

std::vector<f_t> testClasses, trainClasses;

CudaMatrix<f_t> trainInMat, trainOutMat, testInMat, testOutMat, testOutTest;
CudaVector<f_t> result;
CudaVector<f_t> tempVector1, tempVector2;
CudaVector<f_t> constantsGPU, trainOutClasses, testOutClasses;


int main() {
	assert(layers >= 2);

	size_t allocationSize = 0;
	size_t tempAllocationSizeVector = 0;

	calcAllocationSize(allocationSize, tempAllocationSizeVector);
	
	
	Timer t("Allocation", true);
	CudaVector<f_t> All(allocationSize);
	std::cout << "Allocating " << ((allocationSize+ 2* tempAllocationSizeVector) * sizeof(f_t) / 1024.0 / 1024.0) << " MB GPU memory."<<std::endl;
	constantsGPU.initVector(sizeof(constants) / sizeof(constants[0]));
	hostToDevice<f_t>(constantsGPU.getData(), &constants[0], constantsGPU.numElems());
	tempVector1.initVector(tempAllocationSizeVector), tempVector2.initVector(tempAllocationSizeVector);
	All.fill((f_t) 0.0);
	tempVector1.fill((f_t) 0.0);
	tempVector2.fill((f_t) 0.0);
	
	alloc(All, allocationSize);
	cudaDeviceSynchronize();
	t.startNew("Loading Data", true);
	loadData();

	t.startNew("Randomizing", true);
	for (auto w : weights) {
		w.randomize(seed);
	}
	for (auto b : biases) {
		b.randomize(seed);
	}
	cudaDeviceSynchronize();
	t.startNew("Training", true);
	train();
	cudaDeviceSynchronize();
	t.stop();
	CudaMatrix<f_t>::freeCublasHandle();
	return 0;
}

void calculateTestsetError(size_t samples) {
	Timer t("Calculate testset error");
	for (size_t i = 0; i < samples; i++) {
		CudaVector<f_t> out = testOutTest.getColumn(i);
		forwardPropagation(testIn[i], out);
	}
	std::vector<f_t> results(testOutTest.numElems(), (f_t) 0.0);
	deviceToHost<f_t>(&results[0], testOutTest.getData(), testOutTest.numElems());
	size_t right = 0, missed = 0;
	double crossEntropy = 0;
	for (size_t i = 0; i < samples; i++) {
		size_t argmax = 0;
		f_t max = (f_t)0.0;
		size_t expectedClass = (size_t)testClasses[i];
		for (size_t j = 0; j < classes; j++) {
			size_t idx = i * classes + j;
			f_t value = results[idx];
			
			if (value > max) {
				max = value;
				argmax = j;
			}
		}
		f_t value = results[i * classes + expectedClass];
		f_t l = value <= f_t(0.000001) ? -1000 : log(value);
		crossEntropy -= l;
		if (argmax == expectedClass)
			right++;
		else
			missed++;
	}
	double m = (double)missed;
	double testS = (double)samples;
	double misclassificationRate = m / testS;
	double classificationRate = 1.0 - misclassificationRate;
	double meanCrossEntropy = crossEntropy / ((double)samples);

	std::cout << "Miscl. rate: " << misclassificationRate << " mean Cross Entropy: " << meanCrossEntropy << std::endl;
}


void train() {
	
	std::vector <CudaVector<f_t>> trainInputVectors(minibatchSize);
	std::vector <CudaVector<f_t>> trainOutputVectors(minibatchSize);
	for (int i = 0; i <= minibatchIterations; i++) {
		
		for (int m = 0; m < minibatchSize; m++) {
			int next = distrib(gen);
			trainInputVectors[m] = trainIn[next];
			trainOutputVectors[m] = trainOut[next];
		}
		trainMinibatch(trainInputVectors, trainOutputVectors);
		if (i % 50 == 0) {
			std::cout << "MB iteration " << i<< " ";
			calculateTestsetError(10000);
		}
	}
}

void forwardPropagation(CudaVector<f_t>& input, CudaVector<f_t>& output) {
	z[0].copyFrom(input);
	a[0].copyFrom(input);
	for (int i = 1; i < layers; i++) {
		weights[i - 1].mult(a[i - 1], z[i]);
		z[i].add(biases[i - 1], z[i]);
		if (i < layers - 1) {
			z[i].sigmoid(a[i]);
		} else {
			z[i].softMax(a[i], tempVector1);
			output.copyFrom(a[i]);
		}
	}
}

void backwardPropagation(CudaVector<f_t>& output, CudaVector<f_t>& expectedOutput) {
	int i = (int)layers - 2;
	output.sub(expectedOutput, db[i]);
	db[i].mult(a[i], dw[i], false, true);
	for (i--; i >= 0; i--) {
		auto temp1 = tempVector1.getSubvector(0, db[i].numElems() - 1);
		auto temp2 = tempVector2.getSubvector(0, db[i].numElems() - 1);
		weights[i + 1].mult(db[i + 1], temp1, true);
		z[i + 1].sigmoidD(temp2);
		temp1.multElements(temp2, db[i]);
		db[i].mult(a[i], dw[i], false, true);
	}
}

void trainMinibatch(std::vector<CudaVector<f_t>>& inputs, std::vector<CudaVector<f_t>>& expectedOutputs) {
	Timer t("Minibatch Iteration" 
#ifdef _DEBUG 
		,true
#endif
);
	assert(minibatchSize == inputs.size() && minibatchSize == expectedOutputs.size());
	for (int l = 0; l < layers - 1; l++) {
		mbb[l].fill((f_t) 0.0);
		mbw[l].fill((f_t) 0.0);
	}
	for (int i = 0; i < minibatchSize; i++) {
		forwardPropagation(inputs[i], result);
		backwardPropagation(result, expectedOutputs[i]);
		for (int l = 0; l < layers - 1; l++) {
			db[l].add(mbb[l], mbb[l]);
			dw[l].add(mbw[l], mbw[l]);
		}
	}
	for (int l = 0; l < layers - 1; l++) {
		mbb[l].axpy(biases[l], biases[l], constantsGPU.getData() + 1);
		mbw[l].axpy(weights[l], weights[l], constantsGPU.getData() + 1);
	}
	
}

void calcAllocationSize(size_t& allocationSize, size_t& tempAllocationSizeVector) {
	tempAllocationSizeVector = *std::max_element(&nodesPerLayer[0], &nodesPerLayer[0] + layers - 1);
	for (size_t i = 0; i < layers; i++) {
		allocationSize += 2 * nodesPerLayer[i]; //a, z
		if (i < layers - 1) {
			allocationSize += 3 * (nodesPerLayer[i] * nodesPerLayer[i + 1] + nodesPerLayer[i + 1]); // w, dw, mbw, b, db, mbb
		}
	}
	allocationSize += outputSize; // result

	allocationSize += (inputSize + outputSize) * (testDataSize + trainDataSize) + trainDataSize + testDataSize * (outputSize + 1); // trainIn, trainOut, testIn, testOut, trainOutClasses, testOutClasses, testOutTest
}

void alloc(CudaMatrix<f_t>& All, size_t allocationSize) {
	f_t* pos = All.getData();
	for (int i = 0; i < layers; i++) //weights, dw, biases, db, mbw, mbb, a, z
	{
		size_t rows = nodesPerLayer[i + 1], cols = nodesPerLayer[i], mElems = rows * cols;
		if (i < layers - 1) {
			weights.push_back(CudaMatrix<f_t>(pos, rows, cols));
			pos += mElems;
			dw.push_back(CudaMatrix<f_t>(pos, rows, cols));
			pos += mElems;
			mbw.push_back(CudaMatrix<f_t>(pos, rows, cols));
			pos += mElems;
			biases.push_back(CudaVector<f_t>(pos, rows));
			pos += rows;
			db.push_back(CudaVector<f_t>(pos, rows));
			pos += rows;
			mbb.push_back(CudaVector<f_t>(pos, rows));
			pos += rows;
		}
		a.push_back(CudaVector<f_t>(pos, cols));
		pos += cols;
		z.push_back(CudaVector<f_t>(pos, cols));
		pos += cols;
	}
	result.initVector(pos, outputSize);
	pos += outputSize;
	trainInMat.init(pos, inputSize, trainDataSize);
	for (size_t i = 0; i < trainDataSize; i++) {
		trainIn.push_back(trainInMat.getColumn(i));
	}
	pos += trainInMat.numElems();
	trainOutMat.init(pos, outputSize, trainDataSize);
	for (size_t i = 0; i < trainDataSize; i++) {
		trainOut.push_back(trainOutMat.getColumn(i));
	}
	pos += trainOutMat.numElems();
	testInMat.init(pos, inputSize, testDataSize);
	for (size_t i = 0; i < testDataSize; i++) {
		testIn.push_back(testInMat.getColumn(i));
	}
	pos += testInMat.numElems();
	testOutMat.init(pos, outputSize, testDataSize);
	for (size_t i = 0; i < testDataSize; i++) {
		testOut.push_back(testOutMat.getColumn(i));
	}
	pos += testOutMat.numElems();

	testOutTest.init(pos, outputSize, testDataSize);
	pos += testOutTest.numElems();

	testOutClasses.initVector(pos, testDataSize);
	pos += testOutClasses.numElems();

	trainOutClasses.initVector(pos,trainDataSize);
	pos += trainOutClasses.numElems();

	size_t diff = pos - All.getData();
	assert(diff == allocationSize);
}



void loadData() {
	using namespace std;

	size_t n = trainInMat.numElems();
	vector<f_t> data(n);
	readIntoVector("./data/mnist_train_in_60000.txt", data);
	hostToDevice<f_t>(trainInMat.getData(), &data[0], n);

	n = testInMat.numElems();
	data = vector<f_t>(n);
	readIntoVector("./data/mnist_test_in_10000.txt", data);
	hostToDevice<f_t>(testInMat.getData(), &data[0], n);

	n = trainOutClasses.numElems();
	trainClasses = vector<f_t>(n);
	readIntoVector("./data/mnist_train_out_60000.txt", trainClasses);
	hostToDevice<f_t>(trainOutClasses.getData(), &trainClasses[0], n);

	n = trainOutMat.numElems();
	vector<f_t> data2(n);
	classesToOutputs(trainClasses, data2);
	hostToDevice<f_t>(trainOutMat.getData(), &data2[0], n);

	n = testOutClasses.numElems();
	testClasses = vector<f_t>(n);
	readIntoVector("./data/mnist_test_out_10000.txt", testClasses);
	hostToDevice<f_t>(testOutClasses.getData(), &testClasses[0], n);

	n = testOutMat.numElems();
	data2 = vector<f_t>(n);
	classesToOutputs(testClasses, data2);
	hostToDevice<f_t>(testOutMat.getData(), &data2[0], n);
}

void classesToOutputs(std::vector<f_t>& classVec, std::vector<f_t>& outputs) {
	for (int i = 0; i < classVec.size(); i++) {
		f_t c = classVec[i];
		for (int j = 0; j < classes; j++) {
			outputs[10 * i + j] = c == j ? (f_t) 1.0 : (f_t) 0.0;
		}
	}
}

void printAll()
{
	std::cout << "----------------------------------------------------" << std::endl;
	for (int l = 0; l < layers; l++) {
		if (l != 0) {
			std::cout << "z " << l << std::endl;
			z[l].print();
			std::cout << "a " << l << std::endl;
			a[l].print();
		}
		if (l < layers - 1) {
			std::cout << "w " << l << std::endl;
			weights[l].print();
			std::cout << "b " << l << std::endl;
			biases[l].print();
			std::cout << "dw " << l << std::endl;
			dw[l].print();
			std::cout << "db " << l << std::endl;
			db[l].print();
		}
	}
}



void readIntoVector(std::string filename, std::vector<f_t>& vec) {
	using namespace std;

	ifstream file(filename);
	size_t idx = 0;
	assert(file.is_open());
	if (file.is_open()) {
		string line;
		while (getline(file, line)) {			
			f_t a = 0;
			for (int i = 0; i < line.size(); i++) {
				char c[1] = { line[i] };
				a = (f_t) atoi(c);
				vec[idx] = a;
				idx++;
			}
			
		}
		file.close();
	}
	else {
		std::cout << "Couldn't open file!" << std::endl;
	}
}
