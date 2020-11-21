cmake_minimum_required(VERSION 3.14)

project(MLChart LANGUAGES CXX)


set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 COMPONENTS Widgets Charts REQUIRED)

add_library(MLChart
    MLChartFrame.h
    MLChartFrame.cpp
    MLChart.cpp
    MLChart.h
    MLChart.ui)

target_link_libraries(MLChart PUBLIC Qt::Widgets Qt::Charts)

set_property(TARGET MLChart PROPERTY CUDA_ARCHITECTURES OFF)
