#pragma once
#include <cuda_fp16.h>
#define EXECUTE_MACRO_FOR_REAL_TYPES(macro) macro(float) macro(double) macro(half);
#define EXECUTE_MACRO_FOR_ADVANCED_REAL_TYPES(macro) EXECUTE_MACRO_FOR_REAL_TYPES(macro) macro(float2) macro(double2) macro(float3) macro(double3) macro(float4) macro(double4);
#define EXECUTE_MACRO_FOR_ALL_TYPES(macro)  EXECUTE_MACRO_FOR_ADVANCED_REAL_TYPES(macro) macro(int) macro(unsigned int) \
			macro(long long int) macro(unsigned long long int) macro(char) macro(unsigned char) macro(bool) macro(short) macro(unsigned short);