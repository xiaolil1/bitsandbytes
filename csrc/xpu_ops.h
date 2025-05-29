#ifndef xpu_ops_H
#define xpu_ops_H

#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <assert.h>

#include <vector>
#include <functional>

//#include <sycl/sycl.hpp>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

typedef enum DataType_t
{
    //General8bit = 0,
    //FP4 = 1,
    NF4 = 2,
} DataType_t;

template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int workgroup_size, const int n, sycl::queue* stream);

#endif
