#pragma once

#include <QApplication>
#include <QObject>
#include "MLChart.h"

class MLChartFrame{
public: 
	MLChartFrame(int & argc, char * argv[]);
	void addSeries(std::vector<std::tuple<float, float>> & points, std::string name);
	void appendSeries(std::vector<std::tuple<float, float>> & points, std::string name);
	void keepOpenUntilExit();
	
	void update();
	~MLChartFrame();
	bool initialized();
private:
	MLChart * chart = nullptr;
	QApplication * app = nullptr;
};