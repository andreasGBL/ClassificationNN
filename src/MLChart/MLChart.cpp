#include "MLChart.h"
#include "./ui_MLChart.h"
#include <iostream>

#include <limits>

MLChart::MLChart(QWidget * parent)
    : QWidget(parent)
    , ui(new Ui::MLChart)
{
    setWindowFlags(Qt::MSWindowsFixedSizeDialogHint);
    chv = new QtCharts::QChartView();
    QGridLayout layout;
    ui->setupUi(this);
    layout.addWidget(chv);
    this->setLayout(&layout);
    this->show();
    QtCharts::QValueAxis * axisX = new QtCharts::QValueAxis();
    QtCharts::QValueAxis * axisY = new QtCharts::QValueAxis();
    int ticks = 25;
    axisX->setRange(1.0f, (float) ticks);
    axisY->setRange(0.0f, (float) ticks - 1);
    axes.push_back(axisX);
    axes.push_back(axisY);
    axisX->setTickCount(20);
    axisY->setTickCount(20);
    axisY->setTitleText("Value");
    axisX->setTitleText("Iteration");
    axisX->setLabelFormat("%i");
    chv->chart()->addAxis(axisX, Qt::AlignBottom);
    chv->chart()->addAxis(axisY, Qt::AlignLeft);
    chv->setRenderHint(QPainter::Antialiasing);
    chv->chart()->legend()->show();
    chv->chart()->setTitle("Neural Network performance chart");
}

void MLChart::addChart(std::vector<std::tuple<float, float>> & Points, QString name)
{
    QtCharts::QLineSeries * s = new QtCharts::QLineSeries();
    s->setName(name);
    for (auto a : Points)
        s->append(std::get<0>(a), std::get<1>(a));
    series.push_back(s);
    chv->chart()->addSeries(s);
    updateAxes();
}

void MLChart::appendSeries(std::vector<std::tuple<float, float>> & Points, QString name)
{
    bool found = false;
    for (int i = 0; i < series.size(); i++) {
        if (series[i]->name() == name) {
            chv->chart()->removeSeries(series[i]);
            for (auto a : Points)
                series[i]->append(std::get<0>(a), std::get<1>(a));
            chv->chart()->addSeries(series[i]);
            found = true;
        }
        if (series[i]->attachedAxes().size() > 0) {
            series[i]->detachAxis(axes[0]);
            series[i]->detachAxis(axes[1]);
        }
        series[i]->attachAxis(axes[0]);
        series[i]->attachAxis(axes[1]);
    }
    if (!found) {
        addChart(Points, name);
    }
    else {
        updateAxes();
    }
}

MLChart::~MLChart()
{
    delete ui;
    for (auto s : series) {
        delete s;
    }
    for (auto a : axes) {
        delete a;
    }
    delete chv;
}

std::tuple<std::tuple<float, float>, std::tuple<float, float>> MLChart::getMinAndMaxOfSeries(std::vector<QtCharts::QLineSeries *> s)
{
    float minX = std::numeric_limits<float>::max(), minY = minX;
    float maxX = -std::numeric_limits<float>::max(), maxY = maxX;
    for (int n = 0; n < s.size(); n++) {
        auto && vec = s[n]->pointsVector();
        for (int i = 0; i < vec.size(); i++) {
            auto && elem = vec[i];
            const float x = (float)elem.x();
            const float y = (float)elem.y();
            if (x > maxX)
                maxX = x;
            if (y > maxY)
                maxY = y;
            if (x < minX)
                minX = x;
            if (y < minY)
                minY = y;
        }
    }
    return std::tuple<std::tuple<float, float>, std::tuple<float, float>>(std::tuple<float, float>(minX, maxX), std::tuple<float, float>(minY, maxY));
}

void MLChart::updateAxes()
{
    auto && minMax = getMinAndMaxOfSeries(series);
    auto && minMaxX = std::get<0>(minMax);
    auto && minMaxY = std::get<1>(minMax);
    auto && minX = std::get<0>(minMaxX);
    auto && maxX = std::get<1>(minMaxX);
    auto && minY = std::get<0>(minMaxY);
    auto && maxY = std::get<1>(minMaxY);
    auto && axisX = axes[0];
    auto && axisY = axes[1];
    axisX->setRange(std::min(1.0f, minX), std::max((float) axes[0]->tickCount(), maxX));
    axisY->setRange(std::min(0.0f, minY), maxY);
    chv->chart()->update();
    chv->update();
}

