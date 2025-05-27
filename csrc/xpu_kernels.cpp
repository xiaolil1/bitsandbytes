#include "xpu_kernels.h"
//#include "xpu_common.h"


template<typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, /*T *out,*/ const int wgsize) //, /*const int n*/)
{
  std::cout<<"this is kDequantizeBlockwise \n";
}


//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void kDequantizeBlockwise<sycl::half, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, /*sycl::half *out,*/ const int blocksize); //, /*const int n*/);
//template void kDequantizeBlockwise<sycl::half, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, sycl::half *out, const int blocksize, const int n);
template void kDequantizeBlockwise<sycl::half, 512, 64, 8, NF4>(float *code, unsigned char * A, float * absmax, /*sycl::half *out,*/ const int blocksize); //, const int n);
template void kDequantizeBlockwise<float, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, /*float *out,*/ const int blocksize); //, const int n);
//template void kDequantizeBlockwise<float, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n);
template void kDequantizeBlockwise<float, 512, 64, 8, NF4>(float *code, unsigned char * A, float * absmax, /*float *out,*/ const int blocksize); //, const int n);
template void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, /*sycl::ext::oneapi::bfloat16 *out,*/ const int blocksize); //, const int n);
//template void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, sycl::ext::oneapi::bfloat16 *out, const int blocksize, const int n);
template void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, 512, 64, 8, NF4>(float *code, unsigned char * A, float * absmax, /*sycl::ext::oneapi::bfloat16 *out,*/ const int blocksize); //, const int n);


