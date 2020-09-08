cmake_minimum_required(VERSION 3.5)

project(MLChart LANGUAGES CXX)


set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 COMPONENTS Widgets Charts REQUIRED)

add_library(${PROJECT_NAME}
    MLChartFrame.h
    MLChartFrame.cpp
    MLChart.cpp
    MLChart.h
    MLChart.ui)

target_link_libraries(${PROJECT_NAME} PUBLIC Qt::Widgets Qt::Charts)