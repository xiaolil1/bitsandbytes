#include "xpu_kernels.h"
//#include "xpu_common.h"


template<typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, T *out, const int wgsize, const int n)
{
  std::cout<<"this is kDequantizeBlockwise \n";
}


