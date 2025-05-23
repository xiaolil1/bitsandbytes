////placeholder for test build
#include <common.h>
#include <xpu_ops.h>
#include <xpu_kernels.h>


template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int wgsize, const int n, sycl::queue* stream)
{
  std::cout<<"this is dequantizeBlockwise \n";
  kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>(code, A, absmax, out, wgsize, n);
}

