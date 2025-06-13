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
    sycl::nd_range<dim> range,
    sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    cgh.parallel_for<ker_t>(range, ker);
  };
  q.submit(cgf);
  //TODO: Just for debug log print, will be removed.
  //q.wait();
}

template <typename ker_t, int dim>
static inline void sycl_comp_kernel_submit(
    sycl::nd_range<dim> range,
    sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    ker.sycl_ker_local_memory_creation(cgh);
    cgh.parallel_for<ker_t>(range, ker);
  };
  q.submit(cgf);
  //TODO: Just for debug log print, will be removed.
  //q.wait();
}

typedef enum DataType_t
{
    General8bit = 0,
    FP4 = 1,
    NF4 = 2,
} DataType_t;

template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int workgroup_size, const int n, sycl::queue* stream);
template <typename T, int BITS> void gemm_4bit_inference(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize, sycl::queue* stream);

#endif
