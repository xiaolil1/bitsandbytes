#include "xpu_cutlass_2.h"
#include <bit>
#include <cmath>
#include <iostream>

#include <sycl/sycl.hpp>

template <typename T, int BITS>
void gemv_4bit_fusion(int M, int N, int K, T *A, unsigned char *B,
                         float *absmax, float *datatype, T *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream) {
  std::cout<<"this is gemv_4bit fusion path !!!\n";

  size_t idx = item.get_local_id();
  const int sg_idx = idx / SUBG_SIZE;
  const int sg_lane = idx % SUBG_SIZE;
  const int num_values_4bit = SUBG_SIZE;
  const int row_B = NUM_PER_THREAD * item.get_group().get_group_id() + sg_idx;
  const int offset_B = ldb * row_B;
  const int num_values_8bit = num_values_4bit / 2;
  sycl::local_accessor<T, 1> quant_map;

  unsigned char local_B_4bit[num_values_8bit];
  //T local_B[num_values_4bit / 4];
  //T local_A[num_values_4bit / 4];
  T local_absmax = T(0.0f);

  if (idx < 16) {
    quant_map[idx] = T(datatype[idx]);
  }

  item.barrier(sycl::access::fence_space::local_space);

  for (int inner_idx = sg_lane * num_values_4bit; inner_idx < K;
       inner_idx += SUBG_SIZE * num_values_4bit) {
    const int inner_idx_halved = inner_idx / 2;

    // Avoid expensive divsion by the blocksize (as blocksize will always be a
    // power-of-2)
    const int absidx = ((2 * offset_B) + inner_idx) >>
                       (31 - std::countl_zero((unsigned int)blocksize));
    local_absmax = absmax[absidx];

    if (row_B < N) {
      if ((inner_idx_halved + num_values_8bit) < (K / 2)) {
        reinterpret_cast<sycl::vec<int, 4>(&)[num_values_8bit]>(
            local_B_4bit)[0] =
            reinterpret_cast<sycl::vec<int, 4> *>(
                B)[(offset_B + (inner_idx_halved)) / (num_values_8bit)];
      } else {
#pragma unroll
        for (int j = 0; j < (num_values_8bit); j++)
          if ((inner_idx_halved) + j < (K / 2))
            local_B_4bit[j] = B[offset_B + inner_idx_halved + j];
          else
            local_B_4bit[j] = 0b01110111;
      }
    } else {
#pragma unroll
      for (int j = 0; j < (num_values_8bit); j++)
        local_B_4bit[j] = 0b01110111;
    }

    for (int i = 0; i < 4; i++) {
#pragma unroll
      for (int k = 0; k < num_values_8bit / 4; k++) {
        local_B[k * 2] =
            quant_map[local_B_4bit[(i * num_values_8bit / 4) + k] >> 4] *
            local_absmax;
        local_B[k * 2 + 1] =
            quant_map[local_B_4bit[(i * num_values_8bit / 4) + k] & 0x0F] *
            local_absmax;
      }

      if (inner_idx + (num_values_4bit / 4) + (i * num_values_4bit / 4) < K) {
        if (BITS == 16) {
          reinterpret_cast<sycl::vec<int, 4>(&)[num_values_4bit / 4]>(
              local_A)[0] =
              reinterpret_cast<sycl::vec<int, 4> *>(
                  A)[inner_idx / (num_values_4bit / 4) + i];
        } else {
          reinterpret_cast<sycl::vec<int, 4>(&)[num_values_4bit / 4]>(
              local_A)[0] =
              reinterpret_cast<sycl::vec<int, 4> *>(
                  A)[inner_idx / (num_values_4bit / 8) + (2 * i) + 0];
          reinterpret_cast<sycl::vec<int, 4>(&)[num_values_4bit / 4]>(
              local_A)[1] =
              reinterpret_cast<sycl::vec<int, 4> *>(
                  A)[inner_idx / (num_values_4bit / 8) + (2 * i) + 1];
        }

      } else {
#pragma unroll
        for (int k = 0; k < num_values_4bit / 4; k++)
          if (inner_idx + (i * num_values_4bit / 4) + k < K)
            local_A[k] = A[inner_idx + k + (i * num_values_4bit / 4)];
          else
            local_A[k] = T(0.0f);
      }

// accumulate in float for accuracy;
#if 0
#pragma unroll
      for (int k = 0; k < num_values_4bit / 4; k++) {
        local_C += (float)(local_A[k] * local_B[k]);
      }
#else
      //TODO: cute::gemm
#endif      
      local_C = sycl::reduce_over_group(item.get_sub_group(), local_C, sycl::plus<>());
    }
  }

  local_C = sycl::reduce_over_group(item.get_sub_group(), local_C, sycl::plus<>());

  if (row_B < N && sg_lane == 0)
      out[row_B] = T(local_C[row_B]);
}

template void gemv_4bit_fusion<sycl::half, 16>(
    int m, int n, int k, sycl::half *A, unsigned char *B, float *absmax,
    float *datatype, sycl::half *out, int lda, int ldb, int ldc, int blocksize,
    sycl::queue *stream);
//template void gemv_4bit_inference<sycl::ext::oneapi::bfloat16, 16>(
//    int m, int n, int k, sycl::ext::oneapi::bfloat16 *A, sycl::ext::oneapi::bfloat16 *B,
//    float *absmax, float *datatype, float *out, int lda,
//    int ldb, int ldc, int blocksize, sycl::queue *stream);
//template void gemv_4bit_inference<float, 32>(int m, int n, int k, float *A,
//                                             unsigned char *B, float *absmax,
//                                             float *datatype, float *out,
//                                             int lda, int ldb, int ldc,
//                                             int blocksize,
//                                             sycl::queue *stream);
//
