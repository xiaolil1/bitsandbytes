#include <float.h>
#include <xpu_ops.h>

#ifndef xpu_kernels
#define xpu_kernels


//template<typename T, int WG_SIZE, int NUM_PER_TH, int STOCHASTIC, int DATA_TYPE> void kQuantizeBlockwise(float * code, T * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n);

template<typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE> void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, /*T *out,*/ const int wgsize); // /*??*/, const int n);


#endif
