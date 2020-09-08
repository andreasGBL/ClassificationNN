#ifndef MLCHART_H
#define MLCHART_H

#include <QWidget>
#include <QtCharts/QChartView>
#include <QGridLayout>
#include <QtCharts/QLineSeries>
#include <QValueAxis>

QT_BEGIN_NAMESPACE
namespace Ui { class MLChart; }
QT_END_NAMESPACE

class MLChart : public QWidget
{
    Q_OBJECT

public:
    MLChart(QWidget *parent = nullptr);
    void addChart(std::vector<std::tuple<float, float>>& Points, QString name);
    void appendSeries(std::vector<std::tuple<float, float>>& Points, QString name);
    ~MLChart();

private:
    std::vector<QtCharts::QValueAxis*> axes;
    std::tuple<std::tuple<float, float>, std::tuple<float, float>> getMinAndMaxOfSeries(std::vector<QtCharts::QLineSeries *> s);
    void updateAxes();
    Ui::MLChart *ui;
    QtCharts::QChartView* chv;
    std::vector<QtCharts::QLineSeries*> series;
};
#endif // MLCHART_H
