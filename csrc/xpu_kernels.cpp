#include "xpu_kernels.h"
//#include "xpu_common.h"

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
void kDequantizeBlockwise_kernel(
    float* code,
    uint8_t* A,
    float* absmax,
    T* out,
    const int blocksize,
    const int n,
    sycl::nd_item<1>& item) {
  const int base_idx = (item.get_group(0) * TILE_SIZE);
//
//  uint8_t qvals[NUM_PER_TH]; // quantized data
//  T vals[NUM_PER_TH * 2]; // dequantized data
//
//  float* qvals_f = reinterpret_cast<float*>(qvals);
//  float* vals_f = reinterpret_cast<float*>(vals);
//
//  float local_abs_max =
//      absmax[(base_idx + item.get_local_id(0) * NUM_PER_TH) / (blocksize)];
//
//  // load A to qvals
//  float* A_f = reinterpret_cast<float*>(
//      &A[(base_idx + item.get_local_id(0) * NUM_PER_TH)]);
//#pragma unroll
//  for (int j = 0; j < NUM_PER_TH / (sizeof(float) / sizeof(uint8_t)); j++) {
//    qvals_f[j] = A_f[j];
//  }
//
//#pragma unroll
//  for (int j = 0; j < NUM_PER_TH; j++) {
//    // unpack to val and dequant
//    vals[j * 2] =
//        static_cast<T>(dDequantizeNF4<float>(qvals[j] & 0x0F) * local_abs_max);
//    vals[j * 2 + 1] =
//        static_cast<T>(dDequantizeNF4<float>(qvals[j] >> 4) * local_abs_max);
//  }
//
//  // write to output
//  float* out_f = reinterpret_cast<float*>(
//      &out[base_idx * 2 + item.get_local_id(0) * NUM_PER_TH * 2]);
//#pragma unroll
//  for (int j = 0; j < NUM_PER_TH * 2 / (sizeof(float) / sizeof(T)); j++) {
//    out_f[j] = vals_f[j];
//  }
}

template <
    typename T,
    int TILE_SIZE,
    int THREADS,
    int NUM_PER_TH,
    int DATA_TYPE>
struct kDequantizeBlockwiseFunctor {
  void operator()(sycl::nd_item<1> item) const {
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

////just for test//////////
#define init_data() (rand() / double(RAND_MAX))

template<typename T, int DATA_TYPE>
void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, T *out, const int blocksize, const int n, sycl::queue* stream)
{
  std::cout<<"this is kDequantizeBlockwise \n";
  ////at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue();
  //auto& queue = *stream; //stream->queue(); //c10::xpu::getCurrentXPUStream().queue(); //dpcppGetCurrentQueue();

  //const int workgroup_size = 128;
  //const int num_per_th = 4;
  //const int tile_size = workgroup_size * num_per_th;
  //const int workgroup_num = (n + tile_size - 1) / tile_size / 2;
  //sycl::range<1> local_range{workgroup_size};
  //sycl::range<1> global_range{workgroup_num * workgroup_size};

  //kDequantizeBlockwiseFunctor<
  //    T,
  //    tile_size,
  //    workgroup_size,
  //    num_per_th,
  //    DATA_TYPE>kfn(code, A, absmax, out, blocksize / 2, n);

  //sycl_kernel_submit<decltype(kfn), 1>(sycl::nd_range<1>(sycl::range<1>(global_range), sycl::range<1>(local_range)), queue, kfn);

/////////////////////test ////////////////////////
    std::vector<sycl::device> root_devices;
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
       if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
          continue;
       }
       auto device_list = platform.get_devices();
       for (const auto& device : device_list) {
          if (device.is_gpu()) {
             root_devices.push_back(device);
          }
       }
    }
    auto first = root_devices[0];
    std::cout << "[Watch out] Your device" << std::endl;
    std::cout<<"device backend is: "<<first.get_backend()<<std::endl;
    sycl::device machine(first);
    sycl::queue queue(machine, {sycl::property::queue::in_order()});

    unsigned int element_size = 10;

    auto src_h_1 = sycl::aligned_alloc_host<T>(512, element_size, queue);
    auto src_d_1 = sycl::aligned_alloc_device<T>(512, element_size, queue);
    auto dst_h = sycl::aligned_alloc_host<T>(512, 2 * element_size, queue);
    auto dst_d = sycl::aligned_alloc_device<T>(512, 2 * element_size, queue);

    // Prepare Input
    for (int i = 0; i < element_size; ++i)
    {
        src_h_1[i] = static_cast<T>(init_data());
    }
    H2D
    queue.memcpy(src_d_1, src_h_1, element_size * sizeof(T));//.wait();

    //D2D
    queue.submit([&](sycl::handler &h_1)
                          { h_1.parallel_for<class Kernel_1>(sycl::range<1>(sycl::range<1>{element_size}),
                                                         [=](sycl::item<1> item)
                                                         {
                                                             auto id = item.get_linear_id();
                                                             dst_d[id] = src_d_1[id];
                                                         }); });

    //D2H
    queue.memcpy(dst_h, dst_d, 2 * element_size * sizeof(T)).wait();

    queue.wait();
///////////////////////////////////////////test end///////////////////////////////      
  //sycl_queue.submit([&](sycl::handler& cgh) {
  //  kDequantizeBlockwise_kernel<
  //      T,
  //      TILE_SIZE,
  //      THREADS,
  //      NUM_PER_TH,
  //      DATA_TYPE>fn(code, A, absmax, out, blocksize, n, item);
  //  cgh.parallel_for(
  //          sycl::nd_range<1>(
  //              global_range,
  //              local_range,
  //          fn);
  // });
  //sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

//template void kDequantizeBlockwise<sycl::half, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, /*sycl::half *out,*/ const int blocksize); //, /*const int n*/);
//template void kDequantizeBlockwise<sycl::half, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, sycl::half *out, const int blocksize, const int n);
template void kDequantizeBlockwise<sycl::half, NF4>(float *code, unsigned char * A, float * absmax, sycl::half *out, const int blocksize, const int n, sycl::queue* stream);
//template void kDequantizeBlockwise<float, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, /*float *out,*/ const int blocksize); //, const int n);
//template void kDequantizeBlockwise<float, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n);
template void kDequantizeBlockwise<float, NF4>(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n, sycl::queue* stream);
//template void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, /*sycl::ext::oneapi::bfloat16 *out,*/ const int blocksize); //, const int n);
//template void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, sycl::ext::oneapi::bfloat16 *out, const int blocksize, const int n);
template void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, NF4>(float *code, unsigned char * A, float *absmax, sycl::ext::oneapi::bfloat16 *out, const int blocksize, const int n, sycl::queue* stream);


