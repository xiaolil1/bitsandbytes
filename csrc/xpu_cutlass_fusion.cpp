#include "xpu_cutlass.h"
#include <xpu_ops.h>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdint.h>

#include <sycl/sycl.hpp>

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

using namespace cute;
using namespace cutlass;
using namespace cutlass::gemm;

// Define Basic information 
//Weight-only-quant (B)
using MmaType = cutlass::bfloat16_t;
using QuantType = cutlass::uint8_t; //NF4,FP4

using ElementA = MmaType; //bfloat16_t;
using ElementB = QuantType; //cutlass::gemm::collective::detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;

using ElementMMA = ElementA;
using ElementQuant = QuantType;
using NonVoidElementScale = MmaType;

using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;      // data_type of accumulator
using ElementComputeEpilogue = float;  // data_type of epilogue operations
using ElementOutput = float;

using ProblemShape = Shape<int, int, int, int>;

// Design MMA
// Tile_shape: 256x256x32, Atom_size: 8x16x16 -> confirm Tile_shape % Atom_size = 0
// Block_size (CTA size, workgroup_size, threads number per workgroup): Shape<_8, _4, _1> (Stride<_4, _1, _0> -> contiguous)
// Tile size, (CTA task size, Block_size * inner_loop_number, workgroup_size * inner_loop_number) = Tile_shape (dim3: 256*256*32)
// inner_loop_number (Atom numbers per thread): (256/8) * (256/4) * (32/1)
// XE_8x16x16_F32BF16BF16F32_TT -> hardware 指令
// Stride<_4, _1, _0> could be optional?
using TileShape = Shape<_256, _256, _32>;
using TiledMma =
    typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
                                  Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

// Define Mainloop dispatch policy
constexpr int PipelineStages = 1;
using DispatchPolicy = cutlass::gemm::MainloopIntelPVCMixedPrecision<PipelineStages>;
static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize; // sub_group size

// Design Epilogue
using EpilogueDispatchPolicy = cutlass::epilogue::IntelPVCEpilogue;
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<float /*data_type of GEMM output*/, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;
using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
using SharedStorage = FusionCallBacks::SharedStorage;

// Design Scheduler 
// cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler: batch_size=1, 批处理维度（[L, M, N]），[1,1,1] 表示单批处理
using TileScheduler_ = PersistentScheduler; //TileScheduler_;
static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>, "Intel PVC does not support specializing the tile scheduler.");
using ArchTag = typename DispatchPolicy::ArchTag;
using WorkgroupTileShape = TileShape;
using TileScheduler = typename cutlass::gemm::kernel::detail::TileSchedulerSelector<TileScheduler_, ArchTag, WorkgroupTileShape, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
using TileSchedulerArguments = typename TileScheduler::Arguments;
using TileSchedulerParams = typename TileScheduler::Params;

// Define Epilogue
using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
        EpilogueDispatchPolicy,
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
using EpilogueParams = typename CollectiveEpilogue::Params;
using EpilogueArguments = typename CollectiveEpilogue::Arguments;

using ClusterShape = typename DispatchPolicy::ClusterShape;

// Define Copy
using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;

// **XE_2D_U16x32x32_LD_N**：
// U16：每个线程加载 16 个元素。
// 32x32：每个线程块加载 32x32 的分块。
// LD_N：行主序加载（LD_T 为列主序）。
using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
using StrideA = cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>;
using traits_load_A = Copy_Traits<GmemTiledCopyA, StrideA>;
using atom_load_A = Copy_Atom<traits_load_A, ElementA>;
// **shape_div**：将分块形状 BlockShape 除以线程布局 CopyThreadShape，得到每个线程的子块形状。
using val_layout_load_A = decltype(make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{})));
// 生成分块拷贝策略 Copy_A，包含：
// atom_load_A：原子拷贝操作。
// Layout<CopyThreadShape>：线程布局。
// val_layout_load_A：寄存器片段布局
using Copy_A = decltype(make_tiled_copy(atom_load_A{}, Layout<CopyThreadShape>{}, val_layout_load_A{}));

using GmemTiledCopyB = XE_2D_U8x32x32_LD_V;  // U8  (1-byte) block copy for 8bit-B (narrower type)
using StrideB = cutlass::gemm::TagToStrideB_t<cutlass::layout::RowMajor>;
using traits_load_B = Copy_Traits<GmemTiledCopyB, StrideB>;
using atom_load_B = Copy_Atom<traits_load_B, ElementB>;
using val_layout_load_B = decltype(make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{})));
using Copy_B = decltype(make_tiled_copy(atom_load_B{}, Layout<CopyThreadShape>{}, val_layout_load_B{}));

using GmemTiledCopyScale = XE_2D_U16x1x32_LD_N;
using StrideScale = cute::Stride<_1, int64_t, int64_t>; //dynamic stride
using CopyThreadShapeRev = decltype(cute::reverse(CopyThreadShape{}));
using traits_load_scale = Copy_Traits<GmemTiledCopyScale, StrideScale>;
using atom_load_scale = Copy_Atom<traits_load_scale, NonVoidElementScale>;
using val_layout_load_scale = decltype(make_layout(shape_div(typename traits_load_scale::BlockShape{}, CopyThreadShapeRev{}))); 
using Copy_Scale = decltype(make_tiled_copy(atom_load_scale{}, Layout<CopyThreadShapeRev>{}, val_layout_load_scale{}));

using StrideC = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;
using StrideD = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;
using CopyOpG2R = XE_2D_U32x8x16_LD_N;
using CopyOpR2G = XE_2D_U32x8x16_ST_N;
using GmemTiledCopyC = CopyOpG2R;
using GmemTiledCopyD = cute::conditional_t<not cute::is_void_v<ElementD> && not cute::is_void_v<CopyOpR2G>,
                                             CopyOpR2G, XE_2D_U32x8x16_ST_N>;

template <typename T, int BITS>
class kgemm_4bit_inference_cutlass_dequant {
public:
  // Calculate subgroup_tile_shape (reminder: not the same thing with "subgroup_size" in sycl!!)
  static constexpr auto BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr auto BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr auto BLK_K = get<2>(WorkgroupTileShape{});
  
  //Threads number 
  static constexpr auto ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());
  
  static_assert(BLK_M % TiledMma{}.template tile_size_mnk<0>() == 0, "TiledMma permutation size must match block size.");
  static_assert(BLK_N % TiledMma{}.template tile_size_mnk<1>() == 0, "TiledMma permutation size must match block size.");
  static_assert(BLK_K % TiledMma{}.template tile_size_mnk<2>() == 0, "TiledMma permutation size must match block size.");
  
  //sub-tile shape
  static constexpr auto SG_M = ceil_div(BLK_M, ATOM_M);
  static constexpr auto SG_N = ceil_div(BLK_N, ATOM_N);
  static constexpr auto SG_K = ceil_div(BLK_K, ATOM_K);
  using SubgroupTileShape = Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;
  
  //Total Threads number
  static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K; //32
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  // Kernel level shared memory storage
  struct SharedStorage {
    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
    EpilogueTensorStorage epilogue;
  };

  struct Params {
    int m, n, k;
    T* A;
    uint8_t* B;
    float* out;
    float *datatype; //LUT
	  
    ProblemShape problem_shape{};

	  Copy_A tiled_copy_a;
    Copy_B tiled_copy_b;
    Copy_B tiled_copy_b_4bit;
	  Copy_Scale tiled_copy_scale;
    int group_size;
	
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

  static dim3
    get_block_shape() {
      return dim3(MaxThreadsPerBlock, 1, 1);
  }

  static dim3
  get_grid_shape(Params const& params) {
    dim3 grid = TileScheduler::get_tiled_cta_shape_mnl(params.problem_shape, TileShape{}, ClusterShape{});
    if(params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN) {
      return {grid.y, grid.x, grid.z};
    } else {
      return {grid.x, grid.y, grid.z};
    }
  }

  // Helper functions to select packing for conversion
  template <class SrcType,
            class DstType,
            int Cosize>
  struct select_packing { // Naive packing policy
    static constexpr auto value() {
      //if(cute::thread0()) printf("Cosize, sizeof_bits_v<SrcType> = %d, sizeof_bits_v<DstType> = %d, cute::min(sizeof_bits_v<SrcType>, sizeof_bits_v<DstType>) = %d, 32 / cute::min(sizeof_bits_v<SrcType>, sizeof_bits_v<DstType>) = %d\n", Cosize, sizeof_bits_v<SrcType>, sizeof_bits_v<DstType>, cute::min(sizeof_bits_v<SrcType>, sizeof_bits_v<DstType>), 32 / cute::min(sizeof_bits_v<SrcType>, sizeof_bits_v<DstType>));
      return Int<cute::gcd(Cosize, 32 / cute::min(sizeof_bits_v<SrcType>, sizeof_bits_v<DstType>))>{};
    }
  };
 
   /// Utilities to transform A.
  template <class EngineIn,
            class EngineOut, 
            class EngineRef, 
            class EngineScales, 
            class LayoutIn,
            class LayoutOut,
            class LayoutRef,
            class LayoutScales,
            class... Ts>
  CUTLASS_DEVICE
  void dequant(
    Tensor<EngineIn, LayoutIn> const& tCrA_load, 
    Tensor<EngineOut, LayoutOut>& tCrA_mma,
    Tensor<EngineRef, LayoutRef>& A_ref, //mma_A for debug
    Tensor<EngineScales, LayoutScales>& tCrS_input,
    float* quant_map
  ) {

    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);

    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;

#if 1
    auto const& src = tCrA_load(_, _, _);
    //auto src = src_(_, cute::take(src_.size(1)/2), _);
    //auto src = src_(_, _0{size_t(src_.size(1)/2)}, _);
    auto const& dst = tCrA_mma(_, _, _);
    auto pSrc = const_cast<SrcType*>(raw_pointer_cast(src.data()));
    auto pDst = const_cast<DstType*>(raw_pointer_cast(dst.data()));
    //auto pA = const_cast<DstType*>(raw_pointer_cast(A_ref.data()));
    constexpr int num_elements = decltype(size(src))::value / 2;
    for(int i=0; i<num_elements * 2; i++){
      //if(cute::thread0())
      //printf("ThreadIdxX() = %d, i = %d, *(pSrc + i) = %d, *(pA + i*2) = %f, *(pA + i*2+1) = %f\n", ThreadIdxX(), i, static_cast<int>(*(pSrc + i)), static_cast<int>(*(pA + i*2)), static_cast<int>(*(pA + i*2+1)));
    }

  // TODO(Codeplay): (perf) consider replacing `pack` with `num_elements` here - See xe_flash_attn_mma.hpp
    constexpr int pack = 1; //decltype(select_packing<SrcType, DstType, num_elements>::value())::value;
      int src_size = sizeof_bits_v<SrcType>;
      int dst_size = sizeof_bits_v<DstType>;
      //if(cute::thread0()) printf("Cosize = %d, src_size = %d, dst_size = %d\n", num_elements, src_size, dst_size);
    //using Converter = cutlass::NumericArrayConverter<DstType, SrcType, pack, cutlass::FloatRoundStyle::round_to_nearest>;
#if 1    
    for(int i=0; i<num_elements; i++){
      auto src_value = *(pSrc + i);
      //if(cute::thread0()) printf("*(pSrc + i) = %d, src_value = %d\n",static_cast<int>(*(pSrc + i)), static_cast<int>(src_value));
      *(pDst + (2 * i)) = static_cast<DstType>(quant_map[src_value >> 4]);
      *(pDst + (2 * i + 1)) = static_cast<DstType>(quant_map[src_value & 0x0f]);
      //if(cute::thread0())
      //printf("num_elements = %d, i = %d, *(pSrc + i) = %d, *(pSrc + i) >> 4= %d, *(pSrc + i) & 0x0f, quant_map[*(pSrc + i) >> 4] = %f, quant_map[src_value & 0x0f] = %f \n", num_elements, i, static_cast<int>(*(pSrc + i)), static_cast<int>(*(pSrc + i) >> 4), static_cast<int>(*(pSrc + i) & 0x0f), static_cast<int>(quant_map[*(pSrc + i) >> 4]), static_cast<int>(quant_map[src_value & 0x0f]), static_cast<int>(*(pDst + (2 * i))), static_cast<int>(*(pDst + (2 * i + 1))));
    }
#else
    using SrcArray = cutlass::Array<SrcType, pack>;
    using DstArray = cutlass::Array<DstType, pack>;
    constexpr int iters = num_elements / pack;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < iters ; ++i) {
      SrcArray const* pSrcArr = reinterpret_cast<SrcArray const*>(pSrc) + i;
      DstArray* pDstArr = reinterpret_cast<DstArray*>(pDst) + i;
      //*pDstArr = Converter::convert(*pSrcArr);
	//LUT convert
      const SrcArray& arr = *pSrcArr;
      //#pragma unroll
      //for(int j = 0; j < pack / 2; j++){
      for(int j = 0; j < pack; j++){
        //printf("(*pSrcArr)[%d] = %f\n", j, (*pSrcArr)[j]);
        (*pDstArr)[2 * j] = static_cast<DstType>(1);//quant_map[arr[j] >> 4]);
        (*pDstArr)[2 * j + 1] = static_cast<DstType>(1);//quant_map[arr[j] & 0x0f]);
        //if(i==0)
        //if(cute::thread0()) printf("num_elements = %d, pack = %d, i = %d, j = %d, arr[j] = %d, arr[j] >> 4 = %d, (*pDstArr)[j] = %f, quant_map[arr[j] >> 4] = %f, quant_map[arr[j] & 0x0f] = %f\n",num_elements, pack,i, j, static_cast<int>(arr[j]), static_cast<int>(arr[j] >> 4), (*pDstArr)[2 * j], quant_map[arr[j] >> 4], quant_map[arr[j] & 0x0f]);
        //if(cute::thread0()) printf("i = %d, j = %d, pSrc + i * src_size = %d, pSrcArr = %d, arr[j] = %d, arr[j] >> 4 = %d\n",i, j, pSrc + i * src_size, pSrcArr, static_cast<int>(arr[j]), static_cast<int>(arr[j] >> 4));
        //(*pDstArr)[j+1] = static_cast<DstType>(quant_map[(*pSrcArr)[j] & 0x0F]);
      }
    }
#endif
    // 16 x 4 x 2 values for B
    // 16 x 2 of these are same K
    // 4 different scale/zero values per thread, no exchange needed
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 32; ++j) {
        //if(cute::thread0()) printf("tCrA_mma(_, i, _)[j] = %f, i = %d, j = %d, tCrS_input(i) = %f\n",tCrA_mma(_, i, _)[j], i, j, tCrS_input(i));
        tCrA_mma(_, i, _)[j] *= tCrS_input(i);
        //if(cute::thread0()) printf("after scaling tCrA_mma(_, i, _)[j] = %f\n", tCrA_mma(_, i, _)[j]);
      }
    }
#else
    auto const& src = tCrA_load(_, _, _);
    //auto const& dst = tCrA_mma(_, _, _);
    auto pSrc = const_cast<SrcType*>(raw_pointer_cast(src.data()));
    //auto pSrc = const_cast<SrcType*>(src.data());
    //auto pSrc = reinterpret_cast<const uint8_t*>(raw_pointer_cast(src.data()));
    //auto pDst = const_cast<DstType*>(raw_pointer_cast(dst.data()));
    constexpr int num_elements = decltype(size(src))::value;
    //CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < num_elements; i++){
       //tCrA_mma(_, _, _)[i] = static_cast<DstType>(1.f);//quant_map[*(pSrc + i) >> 4]) * tCrS_input(i);
       uint8_t offset = i * sizeof_bits_v<SrcType>;
       tCrA_mma(_, _, _)[i * 2] = static_cast<DstType>(quant_map[*(pSrc + offset) >> 4] * tCrS_input(0));
       tCrA_mma(_, _, _)[i * 2 + 1] = static_cast<DstType>(quant_map[*(pSrc + offset) & 0x0F]) * tCrS_input(0);
       //if(cute::thread0())
       printf("num_elements = %d, *(pSrc + offset) = %d, *(pSrc + offset) >> 4 = %d, quant_map[*(pSrc + offset) >> 4] = %f, tCrS_input(i) = %f, tCrA_mma(_, _, _)[i] = %f\n", num_elements, static_cast<int>(*(pSrc + offset)), static_cast<int>(*(pSrc + offset) >> 4), quant_map[*(pSrc + offset) >> 4], tCrS_input(0), tCrA_mma(_, _, _)[i]);
    }
#endif    
  }
  
  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    if(cute::thread0()) printf("this is fusion kernel...........\n"); 

    int M = params.m;
    int N = params.n;
    int K = params.k;
    T* A = params.A;
    uint8_t* B = params.B;
    float* out = params.out;
    float* datatype = params.datatype;
    auto tiled_copy_a = params.tiled_copy_a;
    auto tiled_copy_b = params.tiled_copy_b;
    auto tiled_copy_b_4bit = params.tiled_copy_b_4bit;
	  auto tiled_copy_scale = params.tiled_copy_scale;

    int L = 1;
    auto problem_size = ProblemShape{M, N, K, L};

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
  	
    int thread_idx = int(ThreadIdxX());

//// Load Dequatize LUT and save to SLM, 16 for 4bits
    float* quant_map = reinterpret_cast<float*>(smem_buf);
    if (thread_idx < 16) {
      quant_map[thread_idx] = datatype[thread_idx];
      printf("quant_map[thread_idx] = %f\n", quant_map[thread_idx]); 
    }
    barrier_wait(1);

//// Get the block level coordinate(indexing) for current block
    auto blk_shape = TileShape{}; //256,256,32
    auto blk_shape_4bit = Shape<_256, _256, _16>{}; //TileShape{}; //256,256,32
    int m_coord, n_coord, l_coord; //block index
    if (params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN) {
      if(cute::thread0()) printf("AlongN !!\n");
      m_coord = BlockIdxY();
      n_coord = BlockIdxX();
      l_coord = BlockIdxZ();
    } else {
      if(cute::thread0()) printf("not AlongN !!\n");
      m_coord = BlockIdxX();
      n_coord = BlockIdxY();
      l_coord = BlockIdxZ();
    }
    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
    if(cute::thread0()) printf("M = %d, N=%d, K=%d, L=%d, m_coord = %d, n_coord = %d, l_coord = %d, BlockIdxX() = %d, BlockIdxY() = %d, BlockIdxZ() = %d\n",M, N, K, L, m_coord, n_coord, l_coord, BlockIdxX(), BlockIdxY(), BlockIdxZ());

    constexpr auto workgroup_shape = WorkgroupTileShape{}; //256, 256, 32 
    constexpr auto subgroup_tile_shape = SubgroupTileShape{}; // number of atom level workgroup: 256/8=32, 256/16=16, 32/16=2  
  
    Tensor mA_mkl = cute::get_pvc_tensor(make_shape(M,K,L)); //coordinate tensor: 0,1,2....
    Tensor mB_nkl = cute::get_pvc_tensor(make_shape(N,K,L)); //coordinate tensor: 0,1,2....
    Tensor mB_nkl_4bit = cute::get_pvc_tensor(make_shape(N,K/2,L)); //coordinate tensor: 0,1,2....
  
    // local_tile: 从全局张量中提取线程块（CTA）级别的局部子块
    // gA: 逻辑视图（无实际内存分配）
    Tensor gA = local_tile(mA_mkl, select<0,2>(blk_shape), make_coord(m_coord,_,l_coord));
    Tensor gB = local_tile(mB_nkl, select<1,2>(blk_shape), make_coord(n_coord,_,l_coord));	
    Tensor gB_4bit = local_tile(mB_nkl_4bit, select<1,2>(blk_shape_4bit), make_coord(n_coord,_,l_coord));	
  
//// Allocate the tiled_mma and the accumulators for the (M,N) subgroup_tile_shape
    TiledMma tiled_mma;
    // partition_fragment_C： 将累加器 C 划分为线程私有的寄存器片段
    // tiled_mma：分块策略（含线程布局）[ThreadsM, ThreadsK]。
    // tile_shape：子块形状 [TileM, TileK]。
    // 输出 FragmentC，每个线程处理 (TileM/ThreadsM, TileK/ThreadsK) 的子区域。
    // 分块需与线程布局兼容。
    // 寄存器片段大小需适配硬件限制。
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); 
    clear(accumulators);

//// Create K slicing tiling iterator and count
    // make_shape(128)：创建一个单维度形状 [128]，等价于数学上的线性数组。
    // idx2crd(0, shape)：将线性索引 0 映射到该形状的逻辑坐标。
    // 对于单维度，坐标直接等于索引值。
    // 使用方式：int k = get<0>(coord);  // k = 0
    // cute::make_coord_iterator(A, B): 生成起始坐标A，步长B的迭代器
    auto k_tile_iter  = cute::make_coord_iterator(idx2crd(0, make_shape(K / 2)), make_shape(K / 2));
    int k_tile_count = ceil_div(K / 2, get<2>(workgroup_shape));
    if(cute::thread0()) printf("k_tile_count = %d\n", k_tile_count);
         
//////Run MainLoop//////
    // thread_idx: linear index (threadIdx.x + blockDim.x * threadIdx.y)
    // thr_copy_A : 当前线程负责的数据子块
    auto thr_copy_A = tiled_copy_a.get_slice(thread_idx); //A 的拷贝分片
    auto thr_copy_B = tiled_copy_b.get_slice(thread_idx);
    auto thr_copy_B_4bit = tiled_copy_b_4bit.get_slice(thread_idx);
	  auto thr_copy_scale = tiled_copy_scale.get_slice(thread_idx);
  
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * DispatchPolicy::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx); //MMA 线程分片
  
    // 获取全局数据的线程分片，Partition global counting tensors for MMA
    // partition_A 是一个将矩阵 A 的逻辑分块（Tile）划分为线程私有的数据片段（Fragment) 的操作，主要用于矩阵乘法（GEMM）核心里线程级的数据分配
    // 将全局矩阵 A 的逻辑分块（gA）划分为当前线程（thr_mma）需要处理的寄存器片段
    // thr_mma：线程的 MMA（矩阵乘累加）分片
    // gA：矩阵 A 的全局或共享内存分块
    // tCgA，一个逻辑张量，表示当前线程负责的寄存器片段, 形状由 TiledMMA 策略决定
    // tCgA: t(tensor) C(compute) gA(globaleA); 
    // tCsA: s (shared memory)
    // tCrA: r (register)
    // 虽然每个线程参与多个 Atom 的计算，但 tCgB 的 shape 是针对单个Atom 的线程分片
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);
    Tensor tCgB_4bit = thr_mma.partition_B(gB_4bit);
	
//// Create fragments：将全局或共享内存中的数据分块转换为适合硬件加速器（如 Tensor Core）计算的寄存器格式
    // make_fragment_layout: 为寄存器片段（Fragment）创建内存布局（Layout），确保数据在寄存器中的排布符合硬件指令（如 Tensor Core）的要求
    // 提取分块形状（tCgA） → 生成寄存器布局（make_fragment_layout） → 创建逻辑张量（make_tensor）
    Tensor mma_A = make_tensor<ElementMMA>(make_fragment_layout(tiled_copy_a, tCgA(_,_,_,0).shape()));
    Tensor mma_B = make_tensor<ElementMMA>(make_fragment_layout(tiled_copy_b, tCgB(_,_,_,0).shape()));

    using FragScaleLayout = Layout<Shape<_2, _2, _1>>; // scale 寄存器分布
    Tensor fragment_scale_input = make_tensor<NonVoidElementScale>(FragScaleLayout{}); // 创建scale 寄存器张量

    // narrow input fragment
    Tensor mma_B_4bit = make_tensor<ElementMMA>(make_fragment_layout(tiled_copy_b_4bit, tCgB_4bit(_,_,_,0).shape()));
    Tensor quant_frag = make_tensor<ElementQuant>(decltype(mma_B_4bit.layout()){});

    static_assert(std::is_same_v<typename decltype(quant_frag)::value_type, ElementQuant>);
    static_assert(std::is_same_v<typename decltype(mma_A)::value_type, ElementMMA>);
    static_assert(std::is_same_v<typename decltype(mma_B)::value_type, ElementMMA>);

//// Retile for copy
    // retile_D: 将数据从一种布局（如共享内存）转换为另一种布局（如寄存器片段），确保数据在寄存器中的排列符合硬件指令（如 Tensor Core）的要求
    // 为什么需要 retile_D？共享内存的布局（如行主序 Stride<_1,_128>）可能与硬件指令（如 Tensor Core 的 8x8 分块）不兼容, 通过 retile_D 将数据重排为寄存器需要的布局（如 Stride<_1,_8>）
    // D（Destination）：数据最终需要适配的布局（通常是寄存器布局）
    // thr_copy_A.retile_D(mma_A): 将线程分片的数据（thr_copy_A）从原始布局（共享内存的行主序）重映射为目标布局（mma_A 的寄存器布局）。
    // frag_copy_A: 数据按 mma_A 的布局重新排列后的寄存器片段
    // code Analyze:
    //   (1) Lambda 表达式 [&](){ ... }()
    //     [&]：捕获当前作用域的所有变量（按引用）。
    //     std::make_pair：返回 frag_copy_A 和 frag_copy_B 的元组。
    //     立即执行：() 表示直接调用该 Lambda。
    //   (2) thr_copy_A.retile_D(mma_A)
    //     作用：将 thr_copy_A 的数据按 mma_A 的布局重排到寄存器。
    //     底层操作：
    //       从共享内存读取数据。
    //       按 mma_A.layout() 的步长（如 Stride<_1,_8>）重新排列。
    //       写入寄存器片段 frag_copy_A。
    //   (3) thr_copy_B_4bit.retile_D(quant_frag)
    //     作用：将 4-bit 量化的 thr_copy_B_4bit 数据解压并按 quant_frag 布局重排。
    //     特殊处理：
    //       4-bit 解压：将每字节的 2 个 4-bit 数值解压为 2 个 8-bit 数值。
    //       布局适配：确保解压后的数据符合 MMA 指令的输入要求（如 int8 或 fp16）。   
    //   (4) 为什么需要 make_pair?: C++ 函数（或 Lambda）只能返回一个值，无法直接返回多个独立对象。 std::pair 或 std::tuple 将多个值封装为单个对象。允许 Lambda 函数通过单一 return 返回多个值。
    auto [frag_copy_A, frag_copy_B] = [&](){
        return std::make_pair(thr_copy_A.retile_D(mma_A), thr_copy_B_4bit.retile_D(quant_frag));
    }();

    Tensor copy_tCrS = thr_copy_scale.retile_D(fragment_scale_input);
    
//// Retile global counting tensors for copies: 
    // retile_D：将数据 物理复制到目标布局（如寄存器）。
    // retile_S：仅生成一个 逻辑视图，不实际移动数据（类似 reinterpret_cast）
    // 生成一个逻辑视图 tAgA，其形状和步长与 tCgA 相同，但数据仍存储在原始位置（共享内存）
    // 共享内存 → retile_S → 逻辑视图 (next step later → 寄存器 (实际复制))
    Tensor tAgA = thr_copy_A.retile_S(tCgA);
    Tensor tBgB = thr_copy_B_4bit.retile_S(tCgB);

//// Prepare for prefetch
    // BLK_M, BLK_N, BLK_K, Num_SGs: Gemm Tile Atom information.
    // tiled_copy_a: Copy Atom information
    // prefetch_selector: 选择适合硬件架构的预取策略
    auto tiled_prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>,Int<BLK_K>>, Num_SGs>(tiled_copy_a);
    auto tiled_prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>,Int<BLK_K>>, Num_SGs>(tiled_copy_b_4bit);
    // get_slice: 获取当前线程负责的预取分片
    auto thr_prefetch_A = tiled_prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = tiled_prefetch_b.get_slice(thread_idx);
    
    // Partition global tile for prefetch
    // partition_S：将全局数据划分为预取分片，生成逻辑视图（不实际移动数据）
    // pAgA 和 pBgB：线程私有的全局内存分片视图，用于后续预取操作
    // code analyze:
    //   (1) 预取（Prefetch）的作用: 隐藏延迟：在计算当前分块时，异步预取下一个分块的数据到缓存或共享内存。
    //   (2) partition_S vs partition_D:
    //      partition_S: 生成逻辑视图（源布局），不实际移动数据
    //      partition_D: 实际复制数据到目标布局（如共享内存→寄存器）
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB_4bit);
  
////
//// Mainloop
////
    // 在矩阵乘法（GEMM）中动态计算每个线程块（CTA）需要处理的数据分块位置
    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord_mnkl;
    m_coord = m_idx * BLK_M + (get_sub_group_id() / ATOM_N) * SG_M; // m_idx * BLK_M：分块在 M 维度的起始全局坐标; get_sub_group_id() / ATOM_N) * SG_M：子组在 M 维度的偏移（用于细粒度并行） 
    n_coord = n_idx * BLK_N + (get_sub_group_id() % ATOM_N) * SG_N; // n_idx * BLK_N：分块在 N 维度的起始全局坐标; (get_sub_group_id() % ATOM_N) * SG_N：子组在 N 维度的偏移
    l_coord = l_idx;

    Tensor copy_iter_s = [&](){
        return make_tensor(make_inttuple_iter(make_coord(n_coord, 0, l_coord)), // 初始坐标：(n_coord, 0, l_coord)，表示从 N 维的 n_coord 开始，K 维从 0 开始
                           make_layout(make_shape(_2{}, _2{}, _1{}, k_tile_count), // 迭代器的逻辑形状：[2, 2, 1, k_tile_count]，表示每次迭代生成 2x2x1 的坐标块，共 k_tile_count 次
                                       make_stride(E<0>{} * _16{}, E<0>{} * _32{}, _0{}, E<1>{} * _1{}))); // 步长 [16, 32, 0, 1]：
                                                                                                           //   E<0>{} * _16{}: 第一维度（N）的步长为 16;
                                                                                                           //   E<0>{} * _32{}：第二维度（K）的步长为 32;
                                                                                                           //   0{}：第三维度（L）的步长为 0（固定）;
                                                                                                           //   E<1>{} * _1{}：第四维度（迭代次数）的步长为 1.
                                                                                                           // E<0>{} 是一个编译时表达式，用于表示步长（Stride）或布局（Layout）中的占位符或动态值
                                                                                                           //   E<N>：一个模板类，表示第 N 维的步长或索引，通常用于动态形状或步长的占位符
                                                                                                           //   E<0>{}：表示第 0 维（最内层维度）的动态步长或索引值，具体值在运行时确定
    }();
#if 0
  #define CUTLASS_ENABLE_DEBUG_PRINTS 1
  #if CUTLASS_ENABLE_DEBUG_PRINTS
  #define PRINT(x) print(#x ": "); print(x); print("\n");
    if (cute::thread0()){
        print("======================= A: \n");
        print("  gA   : "); print(gA);   print("\n");
        print("  tCgA : "); print(tCgA); print("\n");
        print("  tAgA : "); print(tAgA); print("\n");
        print("  mma_A : "); print(mma_A); print("\n");
        print("  frag_copy_A : "); print(frag_copy_A); print("\n");

        print("=====================  B :\n");
        print("  gB : ");   print(gB);   print("\n");
        print("  gB_4bit : ");   print(gB_4bit);   print("\n");
        print("  tCgB : "); print(tCgB); print("\n");
        print("  tCgB_4bit : "); print(tCgB_4bit); print("\n");
        print("  tBgB : "); print(tBgB); print("\n");
        print("  mma_B : "); print(mma_B); print("\n");
        print("  mma_B_4bit : "); print(mma_B_4bit); print("\n");
        print("  frag_copy_B : "); print(frag_copy_B); print("\n");

        print("=====================  Config: \n");
        print("  threads per workgroup : "); print(MaxThreadsPerBlock);  print("\n");
        print("  SubgroupTileShape     : "); print(SubgroupTileShape{}); print("\n");

        print("  tiled_prefetch_a :    "); print(tiled_prefetch_a); print("\n");
        print("  tiled_prefetch_b :    "); print(tiled_prefetch_b); print("\n");
        print("  pAgA :    "); print(pAgA); print("\n");
        print("  pBgB :    "); print(pBgB); print("\n");
      }
  #undef PRINT
  #endif
#endif  

    // crd2idx: 将多维逻辑坐标转换为线性索引
    const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K / 2));
    int prefetch_k = 0;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
      prefetch(tiled_prefetch_a, pAgA(_,_,_,prefetch_k));
      prefetch(tiled_prefetch_b, pBgB(_,_,_,prefetch_k));
    }

    const int k_reload_factor = params.group_size / BLK_K / 2;
    if(cute::thread0()) printf("k_reload_factor = %d\n", k_reload_factor); 

    //CUTLASS_PRAGMA_UNROLL
    for (int k_tile = 0, k = k_start_idx; k_tile < k_tile_count; ++k_tile, ++k, ++prefetch_k) {
      // Copy gmem to rmem for the first k_tile
      copy(tiled_copy_a, tAgA(_,_,_,k), frag_copy_A);
      copy(tiled_copy_b_4bit, tBgB(_,_,_,k), frag_copy_B);

      copy(tiled_copy_scale, copy_iter_s(_, _, _, k_start_idx + (k_tile / k_reload_factor)), copy_tCrS);
      //dequant(quant_frag, mma_B_expanded, fragment_scale_input, quant_map);
      dequant(quant_frag, mma_B, mma_A, fragment_scale_input, quant_map);

      if(prefetch_k < k_tile_count) {
        prefetch(tiled_prefetch_a, pAgA(_,_,_,prefetch_k));
        prefetch(tiled_prefetch_b, pBgB(_,_,_,prefetch_k));
      }
auto pA = const_cast<ElementA*>(raw_pointer_cast(mma_A.data()));
int num_A = decltype(size(mma_A))::value;
for(int i=0; i<num_A; i++) {
  printf("num_A = %d, i = %d, pA = %f\n",num_A, i, *(pA+i));
}
auto pB = const_cast<ElementA*>(raw_pointer_cast(mma_B.data()));
int num_B = decltype(size(mma_B))::value;
for(int i=0; i<num_B; i++) {
  printf("num_B = %d, i = %d, pB = %f\n",num_B, i, *(pB+i));
}
auto pAcc = const_cast<float*>(raw_pointer_cast(accumulators.data()));
int num_Acc = decltype(size(accumulators))::value;
for(int i=0; i<accumulators.size(); i++) {
  printf("num_Acc = %d, i = %d, pAcc = %f\n",num_Acc, i, *(pAcc+i));
}

      cute::gemm(tiled_mma, mma_A, mma_B, accumulators);

for(int i=0; i<num_Acc; i++) {
  printf("after gemm i = %d, pAcc = %f\n", i, *(pAcc+i));
}
    }
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>((char*)nullptr);
    CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
    //auto expanded_problem_size = ProblemShape{M, 2 * N, K, 1};
    auto problem_shape_MNKL = append<4>(problem_size, 1);
    epilogue(
      problem_shape_MNKL,
      subgroup_tile_shape,
      blk_coord_mnkl,
      accumulators,
      tiled_mma,
      thread_idx
    );
  }    
};

template <typename T, int BITS>
void gemm_4bit_inference_cutlass_dequant(int m, int n, int k, T *A, unsigned char *B,
                         float *absmax_, float *datatype, float *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream) {
  std::cout<<"this is gemm_4bit_inference_cutlass_dequant ......................!!!!!!\n";

  sycl::queue q = *stream;
  using GemmKernel = kgemm_4bit_inference_cutlass_dequant<T, BITS>;

  static constexpr int smem_size= 512; // (16 * 32) for quant_map
  int l = 1;

  //TODO(Xiaoli): FIX ME?? auto problem_size = ProblemShape{m, n, k};
  auto problem_size = ProblemShape{m, n, k, l};
  //TODO(Xiaoli): FIX ME
  T* absmax = (T*)absmax_;
  
  // Init Params 
  using Params = GemmKernel::Params;
  Params params;

  params.m = m;
  params.n = n;
  params.k = k;
  params.A = A;
  params.B = B;
  params.out = out;
  params.datatype = datatype;
  params.group_size = blocksize;
 
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  // shape: [m, k, l], stride: [1, k, m*k]
  auto mA_mkl = make_tensor(make_gmem_ptr(A), make_layout(make_shape(m, k, l), stride_A));
  Copy_A tiled_copy_a{Copy_A{}.with(mA_mkl)};

  // make_cute_packed_stride: 根据张量形状自动生成内存步长（Stride）的关键函数，其核心目标是优化内存访问模式以适配硬件指令
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  auto mB_nkl = make_tensor(make_gmem_ptr(B), make_layout(make_shape(n, k, l), stride_B));
  Copy_B tiled_copy_b{Copy_B{}.with(mB_nkl)};

  StrideB stride_B_4bit = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k/2, l));
  auto mB_nkl_4bit = make_tensor(make_gmem_ptr(B), make_layout(make_shape(n, k/2, l), stride_B_4bit));
  Copy_B tiled_copy_b_4bit{Copy_B{}.with(mB_nkl_4bit)};
  
  params.tiled_copy_a = tiled_copy_a;
  params.tiled_copy_b = tiled_copy_b;
  params.tiled_copy_b_4bit = tiled_copy_b_4bit;
 
  const int scale_k = cute::ceil_div(k, blocksize);
  const int dq_mn_size = n;
  // shape: [n, scale_k, 1], stride_S: [1, n, n * scale_k]
  StrideScale stride_S = cutlass::make_cute_packed_stride(StrideScale{}, cute::make_shape(dq_mn_size, scale_k, l));
  auto mScale = make_tensor(
        make_gmem_ptr(absmax), //static_cast<NonVoidElementScale *>(absmax)),
        make_layout(make_shape(n, scale_k, l), stride_S));
  Copy_Scale tiled_copy_scale{Copy_Scale{}.with(mScale)};

  params.tiled_copy_scale = tiled_copy_scale;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  auto problem_shape_MNKL = append<4>(problem_size, 1);
  float alpha=1.0;
  float beta=0.f;
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));
  StrideC stride_D = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));

  params.hw_info = hw_info;
  params.epilogue = CollectiveEpilogue::to_underlying_arguments(problem_size, {{alpha, beta}, nullptr, stride_C, out, stride_D}, nullptr);

  TileSchedulerArguments scheduler{};
  params.scheduler = TileScheduler::to_underlying_arguments(
      problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, scheduler, nullptr);

  params.problem_shape = problem_size;

  // Launch Kernel
  //local_range (workgroup_size): 8*4*1, 1, 1
  //global_range (workgroup_size * inner_loop * workgroup_number): N/256, M/256, K/32 ??
  //tile_size (workgroup_size * inner_loop_number): 256, 256, 32
  //block: workgroup_size
  //grid: workgroup number
  dim3 const block = GemmKernel::get_block_shape();
  dim3 const grid = GemmKernel::get_grid_shape(params);

  const syclcompat::dim3 sycl_block(block.x, block.y, block.z); //workgroup_size: 8*4*1, 1, 1
  const syclcompat::dim3 sycl_grid(grid.x, grid.y, grid.z);     //workgroup_number (problem_size / tile_size): N/256, M/256, K/32 
  printf("Host Grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
  printf("Host Block: (%d, %d, %d)\n", block.x, block.y, block.z);

  auto kernel_props = [] {
      return syclcompat::experimental::kernel_properties{
        sycl::ext::oneapi::experimental::sub_group_size<DispatchPolicy::SubgroupSize>
      };
  }();
  syclcompat::experimental::launch_properties launch_props {
    sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
  };
  syclcompat::experimental::launch_policy policy{
    sycl_grid, sycl_block, launch_props, kernel_props
  };

  auto event = syclcompat::experimental::launch<device_kernel<GemmKernel>>(policy, q, params);
  EventManager::getInstance().addEvent(event);
  //syclcompat::wait();
}

template void gemm_4bit_inference_cutlass_dequant<sycl::ext::oneapi::bfloat16, 16>(
    int m, int n, int k, sycl::ext::oneapi::bfloat16 *A, unsigned char *B,
    float *absmax, float *datatype, float *out, int lda,
    int ldb, int ldc, int blocksize, sycl::queue *stream);

