#include <common.h>
#include "xpu_ops.h"
#include <xpu_kernels.h>

template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int blocksize/*block-quant-size*/, const int n, sycl::queue* stream)
{
  std::cout<<"this is dequantizeBlockwise \n";
  //std::cout<<"this is kDequantizeBlockwise \n";
  auto& queue = *stream;
  const int workgroup_size = 128;
  const int num_per_th = 4;
  const int tile_size = workgroup_size * num_per_th;
  const int workgroup_num = (n + tile_size - 1) / tile_size / 2;
  sycl::range<1> local_range{(size_t)workgroup_size};
  sycl::range<1> global_range{(size_t)workgroup_num * (size_t)workgroup_size};

  kDequantizeBlockwise<
      T,
      tile_size,
      workgroup_size,
      num_per_th,
      DATA_TYPE> kfn(code, A, absmax, out, blocksize / 2, n);

    //queue->parallel_for(sycl::nd_range<1>(sycl::range<1>(global_range), sycl::range<1>(local_range)), kfn);
  sycl_kernel_submit<decltype(kfn), 1>(sycl::nd_range<1>(sycl::range<1>(global_range), sycl::range<1>(local_range)), queue, kfn);
}


//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

//template void dequantizeBlockwise<float, General8bit>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, sycl::queue* stream);
//template void dequantizeBlockwise<float, FP4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, sycl::queue* stream);
template void dequantizeBlockwise<float, NF4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, sycl::queue* stream);

//template void dequantizeBlockwise<sycl::half, General8bit>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n, sycl::queue* stream);
//template void dequantizeBlockwise<sycl::half, FP4>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n, sycl::queue* stream);
//template void dequantizeBlockwise<sycl::half, NF4>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n, sycl::queue* stream);

//template void dequantizeBlockwise<sycl::ext::oneapi::bfloat16, General8bit>(float *code, unsigned char *A, float *absmax, sycl::ext::oneapi::bfloat16 *out, int blocksize, const int n, sycl::queue* stream);
//template void dequantizeBlockwise<sycl::ext::oneapi::bfloat16, FP4>(float *code, unsigned char *A, float *absmax, sycl::ext::oneapi::bfloat16 *out, int blocksize, const int n, sycl::queue* stream);
//template void dequantizeBlockwise<sycl::ext::oneapi::bfloat16, NF4>(float *code, unsigned char *A, float *absmax, sycl::ext::oneapi::bfloat16 *out, int blocksize, const int n, sycl::queue* stream);

