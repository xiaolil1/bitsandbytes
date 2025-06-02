////placeholder for test build
//#include <common.h>
#include "xpu_ops.h"
//#include <xpu_kernels.h>


static const float lookup_table[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f};

template <typename T>
inline T dDequantizeNF4(uint8_t val) {
  return lookup_table[val]; // val < 16
}

template <
    typename T,
    int TILE_SIZE,
    int THREADS,
    int NUM_PER_TH,
    int DATA_TYPE>
SYCL_EXTERNAL void kDequantizeBlockwise_kernel(
    float* code,
    uint8_t* A,
    float* absmax,
    T* out,
    const int blocksize,
    const int n,
    sycl::nd_item<1>& item) {
  const int base_idx = (item.get_group(0) * TILE_SIZE);
  }

  template <
    typename T,
    int TILE_SIZE,
    int THREADS,
    int NUM_PER_TH,
    int DATA_TYPE>
struct kDequantizeBlockwiseFunctor {
SYCL_EXTERNAL void operator()(sycl::nd_item<1> item) const {
    kDequantizeBlockwise_kernel<
        T,
        TILE_SIZE,
        THREADS,
        NUM_PER_TH,
        DATA_TYPE>(code, A, absmax, out, blocksize, n, item);
  }

  kDequantizeBlockwiseFunctor(
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

template<typename T, int DATA_TYPE>
void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, T *out, const int blocksize, const int n, sycl::queue* queue)
//void kdequantizeBlockwise_fp32_nf4(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n, sycl::queue* stream)
{
  std::cout<<"this is kDequantizeBlockwise \n";
  ////at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue();
  //auto& queue = *stream; //stream->queue(); //c10::xpu::getCurrentXPUStream().queue(); //dpcppGetCurrentQueue();
  //sycl::device dev;
  //if (enum_gpu_device(dev)) {
  //  return;
  //}
  //sycl::queue q = sycl::queue(dev, sycl::property_list());

  //int numel = n;

  //float* a = sycl::malloc_shared<float>(numel, q);
  //auto cgf = [&](sycl::handler& cgh) {
  //  cgh.parallel_for<SimpleKer>(sycl::range<1>(numel), SimpleKer(a));
  //};

  //auto e = q.submit(cgf);
  //e.wait();

  //q.memcpy(out, a, numel * sizeof(float));
  //sycl::free(a, q);

  const int workgroup_size = 128;
  const int num_per_th = 4;
  const int tile_size = workgroup_size * num_per_th;
  const int workgroup_num = (n + tile_size - 1) / tile_size / 2;
  sycl::range<1> local_range{(size_t)workgroup_size};
  sycl::range<1> global_range{(size_t)workgroup_num * (size_t)workgroup_size};

queue->submit([&](sycl::handler& cgh) {
  kDequantizeBlockwiseFunctor<
      T,
      tile_size,
      workgroup_size,
      num_per_th,
      DATA_TYPE> kfn(code, A, absmax, out, blocksize / 2, n);
/* queue->submit([&](sycl::handler &h_1)
                       { h_1.parallel_for<>(sycl::range<1>(sycl::range<1>{32}),
                                                      [=](sycl::item<1> item)
                                                      {
                                                          sycl::ext::oneapi::experimental::printf("log1..\n");
                                                      }); });
 queue->wait();*/
  cgh.parallel_for(sycl::nd_range<1>(((n + tile_size - 1) / tile_size) * 64, 64), kfn);
 });
    //queue->parallel_for(sycl::nd_range<1>(sycl::range<1>(global_range), sycl::range<1>(local_range)), kfn);
  //sycl_kernel_submit<decltype(kfn), 1>(sycl::nd_range<1>(sycl::range<1>(global_range), sycl::range<1>(local_range)), queue, kfn);
  }
template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int blocksize/*block-quant-size*/, const int n, sycl::queue* stream)
//void dequantizeBlockwise_fp32_nf4(float *code, unsigned char *A, float *absmax, float *out, int blocksize/*block-quant-size*/, const int n, sycl::queue* stream)
{
  std::cout<<"this is dequantizeBlockwise \n";
  kDequantizeBlockwise<T, DATA_TYPE>(code, A, absmax, out, blocksize, n, stream);
  //kdequantizeBlockwise_fp32_nf4(code, A, absmax, out, blocksize, n, stream);
  //at::xpu::getCurrentXPUStream().queue();

  //sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
  //kDequantizeBlockwise<T, DATA_TYPE>(code /*?*/, A, absmax, out, blocksize, n, stream);
}


//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================
//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

//template void kDequantizeBlockwise<sycl::half, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, /*sycl::half *out,*/ const int blocksize); //, /*const int n*/);
//template void kDequantizeBlockwise<sycl::half, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, sycl::half *out, const int blocksize, const int n);
//template void kDequantizeBlockwise<sycl::half, NF4>(float *code, unsigned char * A, float * absmax, sycl::half *out, const int blocksize, const int n, sycl::queue* stream);
//template void kDequantizeBlockwise<float, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, /*float *out,*/ const int blocksize); //, const int n);
//template void kDequantizeBlockwise<float, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n);
template void kDequantizeBlockwise<float, NF4>(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n, sycl::queue* stream);
//template void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, /*sycl::ext::oneapi::bfloat16 *out,*/ const int blocksize); //, const int n);
//template void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, sycl::ext::oneapi::bfloat16 *out, const int blocksize, const int n);
//template void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, NF4>(float *code, unsigned char * A, float *absmax, sycl::ext::oneapi::bfloat16 *out, const int blocksize, const int n, sycl::queue* stream);


//template void dequantizeBlockwise<float, General8bit>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, sycl::queue* stream);
//template void dequantizeBlockwise<float, FP4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, sycl::queue* stream);
template void dequantizeBlockwise<float, NF4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, sycl::queue* stream);
//template void dequantizeBlockwise<sycl::half, General8bit>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n, sycl::queue* stream);
//template void dequantizeBlockwise<sycl::half, FP4>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n, sycl::queue* stream);
//template void dequantizeBlockwise<sycl::half, NF4>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n, sycl::queue* stream);
//template void dequantizeBlockwise<sycl::ext::oneapi::bfloat16, General8bit>(float *code, unsigned char *A, float *absmax, sycl::ext::oneapi::bfloat16 *out, int blocksize, const int n, sycl::queue* stream);
//template void dequantizeBlockwise<sycl::ext::oneapi::bfloat16, FP4>(float *code, unsigned char *A, float *absmax, sycl::ext::oneapi::bfloat16 *out, int blocksize, const int n, sycl::queue* stream);
//template void dequantizeBlockwise<sycl::ext::oneapi::bfloat16, NF4>(float *code, unsigned char *A, float *absmax, sycl::ext::oneapi::bfloat16 *out, int blocksize, const int n, sycl::queue* stream);

