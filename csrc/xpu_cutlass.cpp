#include "xpu_cutlass.h"
#include <bit>
#include <cmath>
#include <iostream>

#include <sycl/sycl.hpp>

template <typename T, int BITS>
void gemv_4bit_inference(int m, int n, int k, T *A, T *B,
                         float *absmax, float *datatype, float *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream) {
  std::cout<<"this is gemv_4bit_inference cutlass path !!!\n";

//TODO: dispatch for different T

  // Specific setting
  using ElementAccumulator = float;      // data_type of accumulator
  using ElementComputeEpilogue = float;  // data_type of epilogue operations

  //TODO: check the shape/stride size
  // Workgroup-level tile
  using TileShape = Shape<_256, _256, _32>;
  using TiledMma = 
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
                                    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // Create the Epilogue
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<float /*data_type of GEMM output*/, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<cutlass::epilogue::IntelPVCEpilogue, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          cutlass::epilogue::IntelPVCEpilogue,
          TileShape,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>, // Convert CUTLASS 2.x to CUTLASS 3.x representation
          float, // data_type of output: out
          cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>, // Convert CUTLASS 2.x to CUTLASS 3.x representation
          FusionCallBacks,
          XE_2D_U32x8x16_LD_N, // The copy atom used to load matrix C
          void, void,
          XE_2D_U32x8x16_ST_N, // The copy atom used to store matrix D
          void, void>;

  // GEMM Mainloop - iteration over blocks in K dimension
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          cutlass::gemm::MainloopIntelPVC<2>, //use PipelineStages = 2
          TileShape,
          bfloat16_t, // data_type of input: A
          cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>, // Convert CUTLASS 2.x to CUTLASS 3.x representation
          bfloat16_t, // data_type of input: B
          cutlass::gemm::TagToStrideB_t<cutlass::layout::RowMajor>, // Convert CUTLASS 2.x to CUTLASS 3.x representation
          TiledMma,
          XE_2D_U16x32x32_LD_N, // 2D block copy operations used for A
          void, void, cute::identity,
          XE_2D_U16x32x32_LD_V, // 2D block copy operations used for B
          void, void, cute::identity
  >;

  // Define the whole kernel (mainloop and epilogue)
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
          Shape<int, int, int, int>, // Defer global problem shape definition to runtime
          CollectiveMainloop,
          CollectiveEpilogue
  >;

  auto problem_size = typename GemmKernel::ProblemShape{m, n, k, 1};
  auto [M, N, K, L]  = cute::append<4>(problem_size, 1);

  typename GemmKernel::StrideA stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, cute::make_shape(M, K, L));
  typename GemmKernel::StrideB stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, cute::make_shape(N, K, L));
  typename GemmKernel::StrideC stride_C = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, cute::make_shape(M, N, L));
  //typename GemmKernel::StrideD stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, cute::make_shape(M, N, L));

  //
  // Data members
  //
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
#if 0  
  cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
  cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
  cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
  //cutlass::DeviceAllocation<typename CollectiveEpilogue::ElementOutput> block_D;

  block_A.reset(M * K * L);
  block_B.reset(K * N * L);
  block_C.reset(M * N * L);
  //block_D.reset(M * N * L);

  //TODO: check whether need the fake data?
  uint64_t seed = 0;
  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
#endif
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
std::cout<<"log1 ...\n";
  float alpha=1.0; //Maybe will not be used by BNB, just keep it temporarily.
  float beta=0.f; //Maybe will not be used by BNB, just keep it temporarily.
  typename GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
#if 1    
    {(bfloat16_t*)A, stride_A, (bfloat16_t*)B, stride_B},
    {{alpha, beta}, nullptr, stride_C, out, stride_C}, hw_info};
#else
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{alpha, beta}, nullptr, stride_C, block_C.get(), stride_C}, hw_info};
#endif

std::cout<<"log2 ...\n";
  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
std::cout<<"log3 ...\n";
  // Run the GEMM
  CUTLASS_CHECK(gemm_op.run());

std::cout<<"log4 ...\n";
  syclcompat::wait();

  //out = (T*)out;
}

//template void gemv_4bit_fusion<sycl::half, 16>(
//    int m, int n, int k, sycl::half *A, unsigned char *B, float *absmax,
//    float *datatype, sycl::half *out, int lda, int ldb, int ldc, int blocksize,
//    sycl::queue *stream);
template void gemv_4bit_inference<sycl::ext::oneapi::bfloat16, 16>(
    int m, int n, int k, sycl::ext::oneapi::bfloat16 *A, sycl::ext::oneapi::bfloat16 *B,
    float *absmax, float *datatype, float *out, int lda,
    int ldb, int ldc, int blocksize, sycl::queue *stream);
//template void gemv_4bit_inference<float, 32>(int m, int n, int k, float *A,
//                                             unsigned char *B, float *absmax,
//                                             float *datatype, float *out,
//                                             int lda, int ldb, int ldc,
//                                             int blocksize,
//                                             sycl::queue *stream);
//
