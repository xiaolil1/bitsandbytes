#include <float.h>
#include <xpu_ops.h>

#ifndef xpu_kernels
#define xpu_kernels

//template <typename ker_t, int dim>
//static inline void sycl_kernel_submit(
//    sycl::nd_range<1> range,
//    sycl::queue q,
//    ker_t ker) {
//  //std::cout<<"this is sycl_kernel_submit ...\n";
//  auto cgf = [&](::sycl::handler& cgh) {
//    cgh.parallel_for<ker_t>(range, ker);
//  };
//  q.submit(cgf);
//}

//template<typename T, int WG_SIZE, int NUM_PER_TH, int STOCHASTIC, int DATA_TYPE> void kQuantizeBlockwise(float * code, T * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n);

//template<typename T, int DATA_TYPE>  void kDequantizeBlockwise;//(float *code, unsigned char * A, float * absmax, T *out, const int wgsize, const int n, sycl::queue* stream);

template <
    typename T,
    int TILE_SIZE,
    int THREADS,
    int NUM_PER_TH,
    int DATA_TYPE>
class kDequantizeBlockwise {
public:
  
  SYCL_EXTERNAL  void operator()(sycl::nd_item<1> item) const;

  kDequantizeBlockwise(
      float* code_,
      uint8_t* A_,
      float* absmax_,
      T* out_,
      const int blocksize_,
      const int n_)
      : code(code_),
        A(A_),
        absmax(absmax_),
        out(out_),
        blocksize(blocksize_),
        n(n_) {}

 private:
  float* code;
  uint8_t* A;
  float* absmax;
  T* out;
  const int blocksize;
  const int n;
};

#endif
