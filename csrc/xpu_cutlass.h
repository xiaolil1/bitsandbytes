#pragma once

#include <float.h>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include <cute/atom/copy_traits_xe.hpp>

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"

//#include "cutlass/cutlass.h"
//#include "cutlass/gemm/dispatch_policy.hpp"
//#include "cutlass/gemm/gemm.h"
//#include "cutlass/kernel_hardware_info.hpp"
//
//#include "cute/algorithm/functional.hpp"
//#include "cute/atom/mma_atom.hpp"
//#include "cute/algorithm/gemm.hpp"
//#include "cute/tensor_predicate.hpp"

using namespace cute;

#if 0
template <typename T, size_t GROUP_SIZE, size_t NUM_PER_THREAD,
          size_t SUBG_SIZE, int BITS>
class kgemv_4bit_inference_cutlass {
public:
  SYCL_EXTERNAL void operator()(sycl::nd_item<1> item) const;

  kgemv_4bit_inference_cutlass(int M_, int N_, int K_, T *A_, unsigned char *B_,
                       float *absmax_, const float *datatype_, T *out_,
                       int lda_, int ldb_, int ldc_, int blocksize_)
      : M(M_), N(N_), K(K_), A(A_), B(B_), absmax(absmax_), datatype(datatype_),
        out(out_), lda(lda_), ldb(ldb_), ldc(ldc_), blocksize(blocksize_),
        quant_map() {}

  void sycl_ker_local_memory_creation(sycl::handler &cgh) {
    quant_map = sycl::local_accessor<T>(16, cgh);
  }

private:
  int M;
  int N;
  int K;
  T *A;
  unsigned char *B;
  float *absmax;
  const float *datatype;
  T *out;
  int lda;
  int ldb;
  int ldc;
  int blocksize;
  sycl::local_accessor<T> quant_map;
};
#endif

template <typename T, int BITS>
void gemv_4bit_inference(int m, int n, int k, T *A, T *B,
                         float *absmax, float *datatype, float *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream);
