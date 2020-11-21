#pragma once
#include <string>
#include <chrono>
#include <iostream>
class Timer
{
public:
	Timer(std::string name, bool print = false) : name(name), print(print) { start(); }
	~Timer() { stop(); }
	inline void start() {
		stopped = false;
		started = std::chrono::steady_clock::now();
	}
	inline double stop(){
		double time = 0.0;
		if (!stopped) {
			time = ((double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count()) / 1000.0;
			if(print)
				std::cout << "Time for " << name << ": " << time << " ms" << std::endl;
		}
		stopped = true;
		return time;
	}
	inline void startNew(std::string name, bool print = false) {
		stop();
		this->name = name;
		start();
	}
private:
	std::chrono::steady_clock::time_point started;
	std::string name;
	bool stopped = true;
	bool print = false;
};