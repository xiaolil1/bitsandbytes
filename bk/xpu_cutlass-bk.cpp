#include "xpu_cutlass.h"
#include <bit>
#include <cmath>
#include <iostream>

#include <sycl/sycl.hpp>

  // The code section below describes datatype for input, output matrices and computation between
  // elements in input matrices.
  using ElementAccumulator = float;      // <- data type of accumulator
  using ElementComputeEpilogue = float;  // <- data type of epilogue operations
  using ElementInputA = bfloat16_t;      // <- data type of elements in input matrix A
  using ElementInputB = bfloat16_t;      // <- data type of elements in input matrix B
  using ElementOutput = float;           // <- data type of elements in output matrix D

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // The 2D block copy operations used for the A and B matrices
  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

  // Workgroup-level tile
  using TileShape = Shape<_256, _256, _32>;


  // A TiledMMA struct defines a tiling of an MMA atom over M, N and K, combining both additional
  // hardware (sub-groups for Intel PVC) and iterations by each sub-group.
  //
  // The TiledMMAHelper struct defines a specific TiledMMA for a given MMA atom
  // (XE_8x16x16_F32BF16BF16F32_TT), TileShape (<256, 256, 32>) and sub-group layout (8x4x1). The
  // TiledMMA constructed using TiledMMAHelper has the property that each sub-group operates on a
  // single contiguous chunk of the work-group TileShape. For this configuration, this implies that
  // each sub-group operates on a contiguous 32x64x32 chunk (4x4x2 iterations). See
  // 0t_mma_atom.md#TiledMMAs for more info. Sub-groups are arranged row-major (stride 4,1,0) for
  // performance reasons.
  using TiledMma =                    // M=8,N=16,K=16, D=f32,A=bf16,B=bf16,C=f32
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
                                    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // For Intel PVC, PipelineStages defines how many k-blocks ahead to prefetch from A and B.
  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelPVC<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelPVCEpilogue;

  // This is the 'default' epilogue operation (Linear Combination) which performs everything in:
  // (D = alpha * (A*B) + beta * C)
  // aside from the (A*B), which is handled by the GEMM. See 05_pvc_gemm_with_epilogues for more
  // complex epilogue examples.
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  // FusionCallbacks ties the EpilogueOp to an implementation (based on the dispatch
  // policy/architecture) and defines the epilogue arguments.
  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  // GEMM Epilogue - loads & stores C/D matrices, performs epilogue operations & load/stores any
  // auxiliary data required
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC>, // Converts CUTLASS 2.x to CUTLASS 3.x representation
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD>, // Converts CUTLASS 2.x to CUTLASS 3.x representation
          FusionCallBacks,
          XE_2D_U32x8x16_LD_N, // The copy atom used to load matrix C
          void, void,
          XE_2D_U32x8x16_ST_N, // The copy atom used to store matrix D
          void, void>;

  // GEMM Mainloop - iteration over blocks in K dimension
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          ElementInputA,
          cutlass::gemm::TagToStrideA_t<LayoutA>, // Converts CUTLASS 2.x to CUTLASS 3.x representation
          ElementInputB,
          cutlass::gemm::TagToStrideB_t<LayoutB>, // Converts CUTLASS 2.x to CUTLASS 3.x representation
          TiledMma,
          GmemTiledCopyA, void, void, cute::identity,  // A
          GmemTiledCopyB, void, void, cute::identity   // B
  >;

  // Define the whole kernel (mainloop and epilogue)
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  Shape<int, int, int, int>, // Defer global problem shape definition to runtime
  CollectiveMainloop,
  CollectiveEpilogue
  >;

  // The GemmUniversalAdapter wraps the defined GEMM kernel and handles the launch, and e.g.
  // persistent scratch memory if required.
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

using LayoutA = typename Gemm::LayoutA;
using LayoutB = typename Gemm::LayoutB;
using LayoutC = typename Gemm::LayoutC;
using LayoutD = typename Gemm::LayoutD;

using ElementA = typename Gemm::ElementA;
using ElementB = typename Gemm::ElementB;
using ElementAcc = typename Gemm::ElementAccumulator;

using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
using ElementC = typename Gemm::ElementC;
using ElementOutput = typename CollectiveEpilogue::ElementOutput;
using ElementCompute = typename CollectiveEpilogue::ElementCompute;
using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed = 0;

cutlass::DeviceAllocation<ElementA> block_A;
cutlass::DeviceAllocation<ElementB> block_B;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<ElementOutput> block_D;
cutlass::DeviceAllocation<ElementOutput> block_ref_D; // Reference GEMM result for verification

void initialize(const ProblemShapeType& problem_size) {
  auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
  auto [M, N, K, L] = problem_shape_MNKL; 

  // Complete the stride by combining static layout info (StrideA) with runtime size info (M,K,L)
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

  block_A.reset(M * K * L);
  block_B.reset(K * N * L);
  block_C.reset(M * N * L);
  block_D.reset(M * N * L);
  block_ref_D.reset(M * N * L);

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
}

template <typename T, int BITS>
void gemv_4bit_inference(int m, int n, int k, T *A, unsigned char *B,
                         float *absmax, float *datatype, T *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream) {
//std::cout<<"this is gemv_4bit_inference cutlass...\n";
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    ProblemShapeType problem_size = ProblemShapeType{m, n, k, ldb};

    initialize(problem_size);

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{1.f, 0.f}, block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    syclcompat::wait();
}

template void gemv_4bit_inference<sycl::half, 16>(
    int m, int n, int k, sycl::half *A, unsigned char *B, float *absmax,
    float *datatype, sycl::half *out, int lda, int ldb, int ldc, int blocksize,
    sycl::queue *stream);
template void gemv_4bit_inference<sycl::ext::oneapi::bfloat16, 16>(
    int m, int n, int k, sycl::ext::oneapi::bfloat16 *A, unsigned char *B,
    float *absmax, float *datatype, sycl::ext::oneapi::bfloat16 *out, int lda,
    int ldb, int ldc, int blocksize, sycl::queue *stream);
template void gemv_4bit_inference<float, 32>(int m, int n, int k, float *A,
                                             unsigned char *B, float *absmax,
                                             float *datatype, float *out,
                                             int lda, int ldb, int ldc,
                                             int blocksize,
                                             sycl::queue *stream);
