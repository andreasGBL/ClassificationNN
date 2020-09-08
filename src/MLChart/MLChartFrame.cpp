#include "MLChartFrame.h"
#include <QTimer>

MLChartFrame::MLChartFrame(int & argc, char * argv[])
{
	app = new QApplication(argc, argv);
	chart = new MLChart();
	update();
}

void MLChartFrame::update()
{
	app->processEvents();
}


void MLChartFrame::addSeries(std::vector<std::tuple<float, float>> & points, std::string name)
{
	chart->addChart(points, QString::fromStdString(name));
}

void MLChartFrame::appendSeries(std::vector<std::tuple<float, float>> & points, std::string name)
{
	chart->appendSeries(points, QString::fromStdString(name));
}

void MLChartFrame::keepOpenUntilExit()
{
	app->exec();
}

MLChartFrame::~MLChartFrame()
{
	delete chart;
	delete app;
}

bool MLChartFrame::initialized()
{
	return chart != nullptr;
}
