#include <common.h>
#include <xpu_kernels.h>
//#include <xpu_cutlass.h>
#include <xpu_ops.h>

//#include "cutlass/epilogue/collective/default_epilogue.hpp"
//#include "cutlass/epilogue/collective/xe_epilogue.hpp"
//#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
//#include "cutlass/gemm/device/gemm_universal.h"
//#include "cutlass/gemm/device/gemm_universal_adapter.h"
//#include "cutlass/gemm/collective/collective_mma.hpp"
//#include "cutlass/util/GPU_Clock.hpp"
//#include "cutlass/epilogue/dispatch_policy.hpp"
//#include <cute/atom/copy_traits_xe.hpp>
//
//#include <cute/tensor.hpp>
//#include <random>
//
//#include "cutlass/util/command_line.h"
//#include "cutlass/util/device_memory.h"
//#include "cutlass/util/packed_stride.hpp"
//#include "cutlass/util/reference/device/gemm_complex.h"
//#include "cutlass/util/reference/device/tensor_compare.h"
//#include "sycl_common.hpp"
//#include "helper.h"

//#include "cutlass/cutlass.h"
//#include "cutlass/gemm/dispatch_policy.hpp"
//#include "cutlass/gemm/gemm.h"
//#include "cutlass/kernel_hardware_info.hpp"
//
//#include "cute/algorithm/functional.hpp"
//#include "cute/atom/mma_atom.hpp"
//#include "cute/algorithm/gemm.hpp"
//#include "cute/tensor_predicate.hpp"
//
//using namespace cute;

template <typename T, int DATA_TYPE>
void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out,
                         int blocksize, const int n, sycl::queue *stream) {
  auto &queue = *stream;
  const int workgroup_size = 128;
  const int num_per_th = 4;
  const int tile_size = workgroup_size * num_per_th;
  if (DATA_TYPE > 0) {
    const int workgroup_num = (n + tile_size * 2  - 1) / (tile_size * 2);
    sycl::range<1> local_range{(size_t)workgroup_size};
    sycl::range<1> global_range{(size_t)workgroup_num * (size_t)workgroup_size};
    kDequantizeBlockwise<T, tile_size, num_per_th, DATA_TYPE> kfn(
        code, A, absmax, out, blocksize / 2, n);
    sycl_kernel_submit<decltype(kfn), 1, 32>(
        sycl::nd_range<1>(sycl::range<1>(global_range),
                          sycl::range<1>(local_range)),
        queue, kfn);
  } else {
    const int workgroup_num = (n + tile_size - 1) / tile_size;
    sycl::range<1> local_range{(size_t)workgroup_size};
    sycl::range<1> global_range{(size_t)workgroup_num * (size_t)workgroup_size};
    kDequantizeBlockwise<T, tile_size, num_per_th, DATA_TYPE> kfn(
        code, A, absmax, out, blocksize, n);
    sycl_kernel_submit<decltype(kfn), 1, 32>(
        sycl::nd_range<1>(sycl::range<1>(global_range),
                          sycl::range<1>(local_range)),
        queue, kfn);
  }
}

#if 0
template <typename T, int BITS>
void gemv_4bit_inference(int m, int n, int k, T *A, unsigned char *B,
                         float *absmax, float *datatype, T *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream) {

  auto &queue = *stream;

  const size_t GROUP_SIZE = 128; // workgroup_size
  const size_t SUBG_SIZE = 32;   // subgroup_size
  const size_t NUM_PER_THREAD = GROUP_SIZE / SUBG_SIZE;
  size_t workgroup_num = (n + NUM_PER_THREAD - 1) / NUM_PER_THREAD;

  kgemv_4bit_inference<T, GROUP_SIZE, NUM_PER_THREAD, SUBG_SIZE, BITS> kfn(
      m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);

  sycl_comp_kernel_submit<decltype(kfn), 1, SUBG_SIZE>(
      sycl::nd_range<1>(sycl::range<1>(GROUP_SIZE * workgroup_num),
                        sycl::range<1>(GROUP_SIZE)),
      queue, kfn);
}
#endif

#if 0
template <typename T, int BITS>
void gemv_4bit_inference(int m, int n, int k, T *A, unsigned char *B,
                         float *absmax, float *datatype, T *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream) {
  std::cout<<"will go to cutlass!!!!!!!!!!!\n";

  auto &queue = *stream;

  const size_t GROUP_SIZE = 128; // workgroup_size
  const size_t SUBG_SIZE = 32;   // subgroup_size
  const size_t NUM_PER_THREAD = GROUP_SIZE / SUBG_SIZE;
  size_t workgroup_num = (n + NUM_PER_THREAD - 1) / NUM_PER_THREAD;

  kgemv_4bit_inference_cutlass<T, GROUP_SIZE, NUM_PER_THREAD, SUBG_SIZE, BITS>(
      m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);
}
#endif
//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void dequantizeBlockwise<float, General8bit>(
    float *code, unsigned char *A, float *absmax, float *out, int blocksize,
    const int n, sycl::queue *stream);
template void dequantizeBlockwise<float, FP4>(float *code, unsigned char *A,
                                              float *absmax, float *out,
                                              int blocksize, const int n,
                                              sycl::queue *stream);
template void dequantizeBlockwise<float, NF4>(float *code, unsigned char *A,
                                              float *absmax, float *out,
                                              int blocksize, const int n,
                                              sycl::queue *stream);

template void dequantizeBlockwise<sycl::half, General8bit>(
    float *code, unsigned char *A, float *absmax, sycl::half *out,
    int blocksize, const int n, sycl::queue *stream);
template void dequantizeBlockwise<sycl::half, FP4>(
    float *code, unsigned char *A, float *absmax, sycl::half *out,
    int blocksize, const int n, sycl::queue *stream);
template void dequantizeBlockwise<sycl::half, NF4>(
    float *code, unsigned char *A, float *absmax, sycl::half *out,
    int blocksize, const int n, sycl::queue *stream);

template void dequantizeBlockwise<sycl::ext::oneapi::bfloat16, General8bit>(
    float *code, unsigned char *A, float *absmax,
    sycl::ext::oneapi::bfloat16 *out, int blocksize, const int n,
    sycl::queue *stream);
template void dequantizeBlockwise<sycl::ext::oneapi::bfloat16, FP4>(
    float *code, unsigned char *A, float *absmax,
    sycl::ext::oneapi::bfloat16 *out, int blocksize, const int n,
    sycl::queue *stream);
template void dequantizeBlockwise<sycl::ext::oneapi::bfloat16, NF4>(
    float *code, unsigned char *A, float *absmax,
    sycl::ext::oneapi::bfloat16 *out, int blocksize, const int n,
    sycl::queue *stream);

#if 0
template void gemv_4bit_inference<sycl::half, 16>(
    int m, int n, int k, sycl::half *A, unsigned char *B, float *absmax,
    float *datatype, sycl::half *out, int lda, int ldb, int ldc, int blocksize,
    sycl::queue *stream);
template void gemv_4bit_inference<sycl::ext::oneapi::bfloat16, 16>(
    int m, int n, int k, sycl::ext::oneapi::bfloat16 *A, unsigned char *B,
    float *absmax, float *datatype, sycl::ext::oneapi::bfloat16 *out, int lda,
    int ldb, int ldc, int blocksize, sycl::queue *stream);
template void gemv_4bit_inference<float, 32>(int m, int n, int k, float *A,
                                             unsigned char *B, float *absmax,
                                             float *datatype, float *out,
                                             int lda, int ldb, int ldc,
                                             int blocksize,
                                             sycl::queue *stream);
#endif
