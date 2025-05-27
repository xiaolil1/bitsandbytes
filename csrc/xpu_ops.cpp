////placeholder for test build
#include <common.h>
#include <xpu_ops.h>
#include <xpu_kernels.h>


template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, /*T *out,*/ int wgsize, /*const int n,*/ sycl::queue* stream)
{
  std::cout<<"this is dequantizeBlockwise \n";
  kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>(code, A, absmax, /*out,*/ wgsize); //, /*n*/);
}


//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

//template void dequantizeBlockwise<float, General8bit>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, sycl::queue* stream);
template void dequantizeBlockwise<float, FP4>(float *code, unsigned char *A, float *absmax, /*float *out,*/ int blocksize, /*const int n,*/ sycl::queue* stream);
template void dequantizeBlockwise<float, NF4>(float *code, unsigned char *A, float *absmax, /*float *out,*/ int blocksize, /*const int n,*/ sycl::queue* stream);
//template void dequantizeBlockwise<sycl::half, General8bit>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n, sycl::queue* stream);
template void dequantizeBlockwise<sycl::half, FP4>(float *code, unsigned char *A, float *absmax, /*sycl::half *out,*/ int blocksize, /*const int n,*/ sycl::queue* stream);
template void dequantizeBlockwise<sycl::half, NF4>(float *code, unsigned char *A, float *absmax, /*sycl::half *out,*/ int blocksize, /*const int n,*/ sycl::queue* stream);
//template void dequantizeBlockwise<sycl::ext::oneapi::bfloat16, General8bit>(float *code, unsigned char *A, float *absmax, sycl::ext::oneapi::bfloat16 *out, int blocksize, const int n, sycl::queue* stream);
template void dequantizeBlockwise<sycl::ext::oneapi::bfloat16, FP4>(float *code, unsigned char *A, float *absmax, /*sycl::ext::oneapi::bfloat16 *out,*/ int blocksize, /*const int n,*/ sycl::queue* stream);
template void dequantizeBlockwise<sycl::ext::oneapi::bfloat16, NF4>(float *code, unsigned char *A, float *absmax, /*sycl::ext::oneapi::bfloat16 *out,*/ int blocksize, /*const int n,*/ sycl::queue* stream);

