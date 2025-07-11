#include <common.h>
#include <xpu_kernels.h>
#include <xpu_cutlass.h>
#include <xpu_ops.h>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/detail/mma.hpp"
#include "cutlass/cuda_host_adapter.hpp"

#include "cutlass/kernel_launch.h"
#if !defined(__CUDACC_RTC__)
#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#endif // !defined(__CUDACC_RTC__)

// 2.x
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"

// 3.x
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/sycl_event_manager.hpp"

static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
}

template <typename T, int BITS>
void gemv_4bit_inference_cutlass(int m, int n, int k, T *A, unsigned char *B,
                         float *absmax, float *datatype, T *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream) {

//  auto &queue = *stream;
//
//  const size_t GROUP_SIZE = 128; // workgroup_size
//  const size_t SUBG_SIZE = 32;   // subgroup_size
//  const size_t NUM_PER_THREAD = GROUP_SIZE / SUBG_SIZE;
//  size_t workgroup_num = (n + NUM_PER_THREAD - 1) / NUM_PER_THREAD;

  kgemv_4bit_inference_cutlass<T, GROUP_SIZE, NUM_PER_THREAD, SUBG_SIZE, BITS> GemmKernel(
      m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);

  dim3 const block = get_block_shape();
  dim3 const grid = get_grid_shape(params);

  const syclcompat::dim3 sycl_block(block.x, block.y, block.z);
  const syclcompat::dim3 sycl_grid(grid.x, grid.y, grid.z);

  // configure smem size and carveout
  int smem_size = GemmKernel::SharedStorageSize;

  //Status launch_result{ Status::kSuccess };
  //  launch_result = Status::kSuccess;
  cutlass::arch::synclog_setup();

  sycl::queue q = *stream; //stream ? *stream : syclcompat::get_default_queue();
  using namespace syclcompat::experimental;
  if constexpr (cute::is_same_v<DispatchPolicy, MainloopDeviceAgnostic>) {
    auto event = launch<device_kernel<GemmKernel>>(launch_policy{
      sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)}
    }, q, params);
    EventManager::getInstance().addEvent(event);
  } else {
    auto event = launch<device_kernel<GemmKernel>>(launch_policy{
      sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)}
      , kernel_properties{sycl_exp::sub_group_size<DispatchPolicy::SubgroupSize>}
    }, q, params);
    EventManager::getInstance().addEvent(event);
  }
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

//template void gemv_4bit_inference<sycl::half, 16>(
//    int m, int n, int k, sycl::half *A, unsigned char *B, float *absmax,
//    float *datatype, sycl::half *out, int lda, int ldb, int ldc, int blocksize,
//    sycl::queue *stream);
template void gemv_4bit_inference_cutlass<sycl::ext::oneapi::bfloat16, 16>(
    int m, int n, int k, sycl::ext::oneapi::bfloat16 *A, unsigned char *B,
    float *absmax, float *datatype, sycl::ext::oneapi::bfloat16 *out, int lda,
    int ldb, int ldc, int blocksize, sycl::queue *stream);
//template void gemv_4bit_inference<float, 32>(int m, int n, int k, float *A,
//                                             unsigned char *B, float *absmax,
//                                             float *datatype, float *out,
//                                             int lda, int ldb, int ldc,
//                                             int blocksize,
//                                             sycl::queue *stream);
