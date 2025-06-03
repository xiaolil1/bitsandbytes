#include "xpu_kernels.h"
//#include "xpu_common.h"
#include <iostream>
#include <cmath>
#include <unistd.h>

#include <sycl/sycl.hpp>

static const float lookup_table[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f};

template <typename T>
inline T dDequantizeNF4(uint8_t val) {
  return lookup_table[val]; // val < 16
}

template <
    typename T,
    int TILE_SIZE,
    int THREADS,
    int NUM_PER_TH,
    int DATA_TYPE>
SYCL_EXTERNAL void kDequantizeBlockwise_kernel(
    float* code,
    uint8_t* A,
    float* absmax,
    T* out,
    const int blocksize,
    const int n,
    sycl::nd_item<1>& item) {
  const int base_idx = (item.get_group(0) * TILE_SIZE);

  uint8_t qvals[NUM_PER_TH]; // quantized data
  T vals[NUM_PER_TH * 2]; // dequantized data

  float* qvals_f = reinterpret_cast<float*>(qvals);
  float* vals_f = reinterpret_cast<float*>(vals);

  float local_abs_max =
      absmax[(base_idx + item.get_local_id(0) * NUM_PER_TH) / (blocksize)];

  // load A to qvals
  float* A_f = reinterpret_cast<float*>(
      &A[(base_idx + item.get_local_id(0) * NUM_PER_TH)]);
#pragma unroll
  for (int j = 0; j < NUM_PER_TH / (sizeof(float) / sizeof(uint8_t)); j++) {
    qvals_f[j] = A_f[j];
  }

#pragma unroll
  for (int j = 0; j < NUM_PER_TH; j++) {
    // unpack to val and dequant
    vals[j * 2] =
        static_cast<T>(dDequantizeNF4<float>(qvals[j] >> 4) * local_abs_max);
    vals[j * 2 + 1] =
        static_cast<T>(dDequantizeNF4<float>(qvals[j] & 0x0F) * local_abs_max);
  }

  // write to output
  float* out_f = reinterpret_cast<float*>(
      &out[base_idx * 2 + item.get_local_id(0) * NUM_PER_TH * 2]);
#pragma unroll
  for (int j = 0; j < NUM_PER_TH * 2 / (sizeof(float) / sizeof(T)); j++) {
    out_f[j] = vals_f[j];
  }
}

template <
    typename T,
    int TILE_SIZE,
    int THREADS,
    int NUM_PER_TH,
    int DATA_TYPE>
SYCL_EXTERNAL void kDequantizeBlockwise<
    T,
    TILE_SIZE,
    THREADS,
    NUM_PER_TH,
    DATA_TYPE>::operator()(sycl::nd_item<1> item) const {
    kDequantizeBlockwise_kernel<
        T,
        TILE_SIZE,
        THREADS,
        NUM_PER_TH,
        DATA_TYPE>(code, A, absmax, out, blocksize, n, item);
  }

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

//template void kDequantizeBlockwise<sycl::half, FP4>(float *code, unsigned char * A, float * absmax, sycl::half *out, const int blocksize, const int n, sycl::queue* stream);
//template void kDequantizeBlockwise<sycl::half, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, sycl::half *out, const int blocksize, const int n);
//template void kDequantizeBlockwise<sycl::half, NF4>(float *code, unsigned char * A, float * absmax, sycl::half *out, const int blocksize, const int n, sycl::queue* stream);

//template void kDequantizeBlockwise<float, FP4>(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n, sycl::queue* stream);
//template void kDequantizeBlockwise<float, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n);
template class kDequantizeBlockwise<float, 512, 128, 4, NF4>;//(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n, sycl::queue* stream);

//template void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, FP4>(float *code, unsigned char * A, float *absmax, sycl::ext::oneapi::bfloat16 *out, const int blocksize, const int n, sycl::queue* stream);
//template void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, sycl::ext::oneapi::bfloat16 *out, const int blocksize, const int n);
//template void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, NF4>(float *code, unsigned char * A, float *absmax, sycl::ext::oneapi::bfloat16 *out, const int blocksize, const int n, sycl::queue* stream);


