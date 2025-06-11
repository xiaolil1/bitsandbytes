#ifndef xpu_ops_H
#define xpu_ops_H

#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <assert.h>

#include <vector>
#include <functional>

#include <sycl/sycl.hpp>
//#include <CL/sycl.hpp>

template <typename ker_t, int dim>
static inline void sycl_kernel_submit(
    sycl::nd_range<1> range,
    sycl::queue q,
    ker_t ker) {
  //std::cout<<"this is sycl_kernel_submit ...\n";
  auto cgf = [&](::sycl::handler& cgh) {
    cgh.parallel_for<ker_t>(range, ker);
  };
  q.submit(cgf);
}

typedef enum DataType_t
{
    //General8bit = 0,
    //FP4 = 1,
    NF4 = 2,
} DataType_t;

template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int workgroup_size, const int n, sycl::queue* stream);

#endif
