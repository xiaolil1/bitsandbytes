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
#include <cutlass/numeric_types.h>
#include <cutlass/bfloat16.h>

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
using QuantType = cutlass::uint4_t; //NF4,FP4

using ElementA = MmaType; //bfloat16_t;
using ElementB = QuantType; //cutlass::gemm::collective::detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;

using ElementMMA = ElementA;
using ElementQuant = QuantType;
using ElementScale = MmaType; //sycl::ext::oneapi::bfloat16; //MmaType;

using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;      // data_type of accumulator
using ElementComputeEpilogue = float;  // data_type of epilogue operations
using ElementOutput = float;

using ProblemShape = Shape<int, int, int, int>;

using TileShape = Shape<_16, _64, _64>;
using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>, Layout<TileShape>,
                                    Layout<Shape<_1, _2, _1>, Stride<_2, _1, _0>>>::TiledMMA;

using WorkgroupTileShape = TileShape;
static constexpr auto BLK_M = get<0>(WorkgroupTileShape{}); //16
static constexpr auto BLK_N = get<1>(WorkgroupTileShape{}); //64
static constexpr auto BLK_K = get<2>(WorkgroupTileShape{}); //64

//Threads number
static constexpr auto ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape()); //1
static constexpr auto ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape()); //2
static constexpr auto ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape()); //1

static_assert(BLK_M % TiledMma{}.template tile_size_mnk<0>() == 0, "TiledMma permutation size must match block size.");
static_assert(BLK_N % TiledMma{}.template tile_size_mnk<1>() == 0, "TiledMma permutation size must match block size.");
static_assert(BLK_K % TiledMma{}.template tile_size_mnk<2>() == 0, "TiledMma permutation size must match block size.");

//sub-tile shape
static constexpr auto SG_M = ceil_div(BLK_M, ATOM_M); //16
static constexpr auto SG_N = ceil_div(BLK_N, ATOM_N); //32
static constexpr auto SG_K = ceil_div(BLK_K, ATOM_K); //64
using SubgroupTileShape = Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>; //<16, 32, 64>

//Total Threads number
static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K; //2
static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{}); //1*2*1*16=32

// Define Mainloop dispatch policy
constexpr int PipelineStages = 3;
using DispatchPolicy = cutlass::gemm::MainloopIntelPVCMixedPrecision<PipelineStages>;
static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize; // sub_group size

// Design Epilogue
using EpilogueDispatchPolicy = cutlass::epilogue::IntelPVCEpilogue;
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<float /*data_type of GEMM output*/, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;
using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
using SharedStorage = FusionCallBacks::SharedStorage;

// Design Scheduler 
using TileScheduler_ = PersistentScheduler; //TileScheduler_;
static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>, "Intel PVC does not support specializing the tile scheduler.");
using ArchTag = typename DispatchPolicy::ArchTag;
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
using CopyThreadShapeRev = decltype(cute::reverse(CopyThreadShape{}));

using GmemTiledCopyA = XE_2D_U16x16x32_LD_N;
using StrideA = cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>;
//using Copy_A = typename Copy_Traits<GmemTiledCopyA, StrideA>::template DefaultTiledCopy<ElementA>;
using traits_load_A = Copy_Traits<GmemTiledCopyA, StrideA>;
using atom_load_A = Copy_Atom<traits_load_A, ElementA>;
using val_layout_load_A = decltype(make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{})));
using Copy_A = decltype(make_tiled_copy(atom_load_A{}, Layout<CopyThreadShape>{}, val_layout_load_A{}));

using GmemTiledCopyB = XE_2D_U4x32x16_LD_T; 
using StrideB = cutlass::gemm::TagToStrideB_t<cutlass::layout::ColumnMajor>;
//using Copy_B = typename Copy_Traits<GmemTiledCopyB, StrideB>::template DefaultTiledCopy<ElementB>;
using traits_load_B = Copy_Traits<GmemTiledCopyB, StrideB>;
using atom_load_B = Copy_Atom<traits_load_B, ElementB>;
using val_layout_load_B = decltype(make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{})));
using Copy_B = decltype(make_tiled_copy(atom_load_B{}, Layout<CopyThreadShape>{}, val_layout_load_B{}));

using GmemTiledCopyScale = XE_2D_U16x1x16_LD_N; 
static constexpr auto SG_QNT_WIDTH = Int<SG_N>{};
using StrideScale = cute::Stride<_1, int64_t, int64_t>; //dynamic stride
using traits_load_scale = Copy_Traits<GmemTiledCopyScale, StrideScale>;
using atom_load_scale = Copy_Atom<traits_load_scale, ElementScale>;
using val_layout_load_scale = decltype(make_layout(shape_div(typename traits_load_scale::BlockShape{}, CopyThreadShapeRev{}))); 
using Copy_Scale = decltype(make_tiled_copy(atom_load_scale{}, Layout<CopyThreadShapeRev>{}, val_layout_load_scale{})); //group-wise scale

using StrideC = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;
using StrideD = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;


template <typename T, int BITS>
class kgemm_4bit_inference_cutlass_dequant {
public:
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
    T *absmax;
    float *datatype; //LUT
    int group_size;
	  
    ProblemShape problem_shape{};

  	Copy_A tiled_copy_a;
    Copy_B tiled_copy_b;
	  Copy_Scale tiled_copy_scale;

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

/*float bfloat16_to_float(uint16_t bf16_bits) {
    uint32_t float_bits = (bf16_bits << 16);  // 将 bfloat16 左移16位转为 float
    return reinterpret_cast<float&>(float_bits);
}*/

   /// Utilities to transform A.
  template <class EngineIn,
            class EngineOut, 
            //class EngineScales, 
            class LayoutIn,
            class LayoutOut,
            //class LayoutScales,
            class... Ts>
  CUTLASS_DEVICE
  void dequant(
    Tensor<EngineIn, LayoutIn> const& in, 
    Tensor<EngineOut, LayoutOut>& out,
    //Tensor<EngineScales, LayoutScales>& tCrS_input,
    float* quant_map
  ) {
    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);

    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;
    //using ScaleType = typename EngineScales::value_type;
#if 1
    int numbers = decltype(size(in))::value;
    for(int i=0; i<numbers; i++){
      //auto in_ptr_8 = (uint8_t*)(raw_pointer_cast(in.data()));
      //out[i] = static_cast<DstType>(quant_map[in_ptr_8[i].data()]);
      uint8_t value = in[i].get();
      out[i] = static_cast<DstType>(quant_map[value]);
      if(cute::thread0()) printf("thread_idx = %d, i = %d, value_bit = %x, value = %d, quant_map[value] = %f, out[i] = %f\n",int(ThreadIdxX()), i, value, static_cast<int>(value), quant_map[value], static_cast<float>(out[i]));
    }
#else    
    static constexpr auto N = decltype(size<1>(in))::value;

    using format_type = ushort; //16
    static constexpr auto src_bits = sizeof_bits_v<SrcType>; //4
    static constexpr auto scalar = sizeof_bits_v<format_type> / src_bits; // 4
    static constexpr auto loop_cnt = decltype(size(out))::value / N; // 128 / 2 = 64
    static_assert((scalar % N) == 0);

    // for tuning performance
    static constexpr auto vec_size = scalar;
    static constexpr auto splits = loop_cnt / vec_size; // 64 / 4 = 16
    static_assert(vec_size <= scalar);

    // reshape tensors for easy access
    auto s_tensor = make_tensor((format_type*)(raw_pointer_cast(in.data())), Shape<Int<loop_cnt / scalar>, Int<N>>{});
    auto d_tensor = make_tensor(out.data(), Shape<Int<vec_size>, Int<splits>, Int<N>>{});

if(cute::thread0())
  printf("thread_idx = %d, decltype(size(in))::value = %d, K = %d, N = %d, L = %d, src_bits = %d, sizeof_bits_v<format_type> = %d, scalar = %d, decltype(size(out))::value = %d, loop_cnt = %d, splits = %d\n",int(ThreadIdxX()), decltype(size(in))::value, decltype(size<0>(in))::value, N, decltype(size<2>(in))::value, src_bits, sizeof_bits_v<format_type>, scalar, decltype(size(out))::value, loop_cnt, splits);

    for (int n = 0; n < N; n++) {
      //const auto ts = tCrS_input(n);

      auto& src = *(cute::array<format_type, loop_cnt / scalar>*)(s_tensor(_, n).data());

      for (int s = 0; s < splits; s++) {
        auto idx =  vec_size * s / scalar;
        auto format_data = src[idx];

        auto& dst = *(cute::array<DstType, vec_size>*)(d_tensor(_, s, n).data());

        for (int i = 0; i < vec_size; i++) {
          uint8_t value = (format_data >> (src_bits * i)) & 0xf;
          dst[i] = (static_cast<DstType>(quant_map[value]));// * ts;          
          //if(cute::thread0())
          printf("n = %d, s = %d, i = %d, src = %d, dst = %f\n", n, s, i, static_cast<int>(value), static_cast<float>(dst[i]));
        }
      }
    }
#endif    
  }
  
  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    if(cute::thread0()) printf("this is fusion kernel...........\n"); 

    int M = params.m;
    int N = params.n;
    int K = params.k;
    int L = 1;

    T* A = params.A;
    uint8_t* B = params.B;
    float* out = params.out;
    float* datatype = params.datatype;

    auto tiled_copy_a = params.tiled_copy_a;
    auto tiled_copy_b = params.tiled_copy_b;
	  //auto tiled_copy_scale = params.tiled_copy_scale;

    auto problem_size = ProblemShape{M, N, K, L};

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

//// Get TID
    int thread_idx = int(ThreadIdxX());
//// Load Dequatize LUT and save to SLM, 16 for 4bits
    float* quant_map = reinterpret_cast<float*>(smem_buf);
    if (thread_idx < 16) {
      quant_map[thread_idx] = datatype[thread_idx];
      //printf("quant_map[thread_idx] = %f\n", quant_map[thread_idx]); 
    }
    barrier_wait(1);

//// Get the block level coordinate(indexing) for current block
    auto blk_shape = TileShape{}; //256,256,32
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
    if(cute::thread0()) {
      printf("M = %d, N=%d, K=%d, L=%d\n", M, N, K, L);
    //}
      printf("thread_idx = %d, m_coord = %d, n_coord = %d, l_coord = %d, BlockIdxX() = %d, BlockIdxY() = %d, BlockIdxZ() = %d\n",thread_idx, m_coord, n_coord, l_coord, BlockIdxX(), BlockIdxY(), BlockIdxZ());

    }
    constexpr auto workgroup_shape = WorkgroupTileShape{}; //256, 256, 32 
    constexpr auto subgroup_tile_shape = SubgroupTileShape{}; //32, 64, 32 (number of atom level workgroup: 256/8=32, 256/4=64, 32/2=32)
  
    Tensor mA_mkl = cute::get_pvc_tensor(make_shape(M,K,L)); //coordinate tensor: 0,1,2....
    Tensor mB_nkl = cute::get_pvc_tensor(make_shape(N,K,L)); //coordinate tensor: 0,1,2....
  
    Tensor gA = local_tile(mA_mkl, select<0,2>(blk_shape), make_coord(m_coord,_,l_coord));
    Tensor gB = local_tile(mB_nkl, select<1,2>(blk_shape), make_coord(n_coord,_,l_coord));		
  
//// Allocate the tiled_mma and the accumulators for the (M,N) subgroup_tile_shape
    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); 
    clear(accumulators);

//// Create K slicing tiling iterator and count
    auto k_tile_iter  = cute::make_coord_iterator(idx2crd(0, make_shape(K)), make_shape(K));
    int k_tile_count = ceil_div(K, get<2>(workgroup_shape)); //inner_loop number
    if(cute::thread0()) printf("k_tile_count = %d\n", k_tile_count);


////// MainLoop //////
    auto thr_copy_A = tiled_copy_a.get_slice(thread_idx);
    auto thr_copy_B = tiled_copy_b.get_slice(thread_idx);
	  //auto thr_copy_scale = tiled_copy_scale.get_slice(thread_idx);
  
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * DispatchPolicy::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx); 
  
//// Create fragments：将全局或共享内存中的数据分块转换为适合硬件加速器（如 Tensor Core）计算的寄存器格式
    // 提取分块形状（tCgA） → 生成寄存器布局（make_fragment_layout） → 创建逻辑张量（make_tensor）
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);
	
    Tensor mma_A = make_tensor<ElementMMA>(make_fragment_layout(tiled_copy_a, tCgA(_,_,_,0).shape()));
    Tensor mma_B = make_tensor<ElementMMA>(make_fragment_layout(tiled_copy_b, tCgB(_,_,_,0).shape()));

	  Tensor dequant_frag = make_tensor<ElementB>(mma_B.layout());
	
    //static constexpr auto scale_traits_size = decltype(size(typename GmemTiledCopyScale::BlockShape{}))::value / SubgroupSize;
    //static constexpr auto scale_traits_num = SG_QNT_WIDTH / size<1>(typename GmemTiledCopyScale::BlockShape{});
    //using FragScaleLayout = Layout<Shape<Int<scale_traits_size>, Int<scale_traits_num>, _1>>;
    //Tensor fragment_scale = make_tensor<ElementScale>(FragScaleLayout{});

    static_assert(std::is_same_v<typename decltype(dequant_frag)::value_type, ElementQuant>);
    static_assert(std::is_same_v<typename decltype(mma_A)::value_type, ElementMMA>);
    static_assert(std::is_same_v<typename decltype(mma_B)::value_type, ElementMMA>);

//// Retile for copy
    Tensor frag_copy_A = thr_copy_A.retile_D(mma_A);
    Tensor frag_copy_B = thr_copy_B.retile_D(dequant_frag);
    //Tensor frag_copy_Scale = thr_copy_scale.retile_D(fragment_scale);
    
//// Retile global counting tensors for copies: 
    Tensor tAgA = thr_copy_A.retile_S(tCgA);
    Tensor tBgB = thr_copy_B.retile_S(tCgB);

//// Prepare for prefetch
    auto tiled_prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>,Int<BLK_K>>, Num_SGs>(tiled_copy_a);;
    auto tiled_prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>,Int<BLK_K>>, Num_SGs>(tiled_copy_b);;
    auto thr_prefetch_A = tiled_prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = tiled_prefetch_b.get_slice(thread_idx);

    // Partition global tile for prefetch
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);
	
//// Run mainloop
//    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord_mnkl;
//    const int n_coord_s = n_idx * BLK_N + (get_sub_group_id() % ATOM_N) * SG_N;
//    const int l_coord_s = l_idx;
//
//    auto copy_iter_s = [&](){
//        return make_tensor(make_inttuple_iter(make_coord(n_coord_s, 0, l_coord_s)),
//                          make_layout(make_shape(Int<scale_traits_size>{}, Int<scale_traits_num>{}, _1{}, k_tile_count),
//                                      make_stride(E<0>{} * _16{}, E<0>{} * size<1>(typename GmemTiledCopyScale::BlockShape{}), _0{}, E<1>{} * _1{})));
//      
//    }();

  #define PRINT(x) print(#x ": "); print(x); print("\n");
    if (cutlass::thread(LOG_THREAD, LOG_GROUP)) {
        print("======================= A: \n");
        print("  gA   : "); print(gA);   print("\n");
        print("  tCgA : "); print(tCgA); print("\n");
        print("  tAgA : "); print(tAgA); print("\n");
        print("  mma_A : "); print(mma_A); print("\n");
        print("  frag_copy_A : "); print(frag_copy_A); print("\n");

        print("=====================  B :\n");
        print("  gB : ");   print(gB);   print("\n");
        print("  tCgB : "); print(tCgB); print("\n");
        print("  tBgB : "); print(tBgB); print("\n");
        print("  mma_B : "); print(mma_B); print("\n");
        print("  frag_copy_B : "); print(frag_copy_B); print("\n");
        print("  dequant_frag : "); print(dequant_frag); print("\n");

        print("=====================  Config: \n");
        print("  threads per workgroup : "); print(MaxThreadsPerBlock);  print("\n");
        print("  SubgroupTileShape     : "); print(SubgroupTileShape{}); print("\n");

        print("  tiled_prefetch_a :    "); print(tiled_prefetch_a); print("\n");
        print("  tiled_prefetch_b :    "); print(tiled_prefetch_b); print("\n");
        print("  pAgA :    "); print(pAgA); print("\n");
        print("  pBgB :    "); print(pBgB); print("\n");
      }
  #undef PRINT
  
	const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K));
    int prefetch_k = k_start_idx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
      prefetch(tiled_prefetch_a, pAgA(_,_,_,prefetch_k));
      prefetch(tiled_prefetch_b, pBgB(_,_,_,prefetch_k));
    }

    for (int k_tile = k_start_idx; k_tile < k_tile_count + k_start_idx; k_tile++, prefetch_k++) {
      barrier_arrive(2);

      // Copy gmem to rmem for the first k_tile
      copy(tiled_copy_a, tAgA(_,_,_,k_tile), frag_copy_A);
      copy(tiled_copy_b, tBgB(_,_,_,k_tile), frag_copy_B);

      const int k_reload_factor = params.group_size / BLK_K;

      //copy(tiled_copy_scale, copy_iter_s(_, _, _, k_tile / k_reload_factor), frag_copy_Scale);

      if(prefetch_k < k_tile_count) {
        prefetch(tiled_prefetch_a, pAgA(_,_,_,prefetch_k));
      }
      if(prefetch_k < k_tile_count / 2) {
        prefetch(tiled_prefetch_b, pBgB(_,_,_,prefetch_k));
      }

      dequant(dequant_frag, mma_B, /*fragment_scale,*/ quant_map);

      cute::gemm(tiled_mma, mma_A, mma_B, accumulators);

//// 在调用gemm前后添加打印逻辑
//auto debug_print = [&](const char* name, auto& tensor) {
//    if (thread_idx == 0) {
//        printf("----- %s -----\n", name);
//        for (int i = 0; i < size<0>(tensor); ++i) {
//            for (int j = 0; j < size<1>(tensor); ++j) {
//                printf("%6.2f ", static_cast<float>(tensor(i, j)));
//            }
//            printf("\n");
//        }
//    }
//    barrier_wait(2);
//};
//
//// 打印输入
//debug_print("Input A (mma_A)", mma_A);
//debug_print("Input B (mma_B)", mma_B);
//debug_print("Accumulators (Before GEMM)", accumulators);
//
//// 执行GEMM
//cute::gemm(tiled_mma, mma_A, mma_B, accumulators);
//
//// 打印输出
//debug_print("Accumulators (After GEMM)", accumulators);

      barrier_wait(2);
    }
	
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>((char*)nullptr);
    CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
    auto problem_shape_MNKL = problem_size; //append<4>(problem_size, 1);
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
void gemm_4bit_inference_cutlass_dequant(int m, int n, int k_, T *A, unsigned char *B,
                         T *absmax_, float *datatype, float *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream) {
  std::cout<<"this is gemm_4bit_inference_cutlass_dequant ......................!!!!!!\n";

int k = k_;


  sycl::queue q = *stream;
  using GemmKernel = kgemm_4bit_inference_cutlass_dequant<T, BITS>;

  static constexpr int smem_size= 512; // (16 * 32) for quant_map
  int l = 1;

  //TODO(Xiaoli): FIX ME?? auto problem_size = ProblemShape{m, n, k};
  auto problem_size = ProblemShape{m, n, k, l};
  //TODO(Xiaoli): FIX ME
//  T* absmax = (T*)absmax_;
//  T* absmax = (T*)absmax_;

#if 1
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
  auto mA_mkl = make_tensor(make_gmem_ptr(A), make_layout(make_shape(m, k, l), stride_A));
  Copy_A tiled_copy_a{Copy_A{}.with(mA_mkl)};

  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
//  auto stride_B_custom = cute::make_stride(
//    cute::Int<1>{},                          // 连续维度步幅（字节）
//    (n * 4 + 7) / 8,            // pitch = ceil(n * 4bit / 8bit)
//    cute::Int<0>{}                           // 无批量步幅（根据需求调整）
//);
//    constexpr int stride_k = (n * 4 ) / 8;
//    constexpr int stride_l = (n * k * 4 ) / 8;
//    auto stride_B = cute::make_stride(
//        cute::Int<1>{},
//        (n * 4 ) / 8,
//        (n * k * 4 ) / 8
//    );
  auto mB_nkl = make_tensor(cute::subbyte_iterator<uint4_t>(B), make_layout(make_shape(n, k, l), stride_B));
  Copy_B tiled_copy_b{Copy_B{}.with(mB_nkl)};

  params.tiled_copy_a = tiled_copy_a;
  params.tiled_copy_b = tiled_copy_b;

  const int scale_k = cute::ceil_div(k, blocksize);
  StrideScale stride_S = cutlass::make_cute_packed_stride(StrideScale{}, cute::make_shape(n, scale_k, l));
  std::cout<<"n = "<<n<<" k = "<<k<<" blocksize = "<<blocksize<<" scale_k = "<<scale_k<<std::endl;

  auto mScale = make_tensor(
        make_gmem_ptr(absmax_), //static_cast<ElementScale *>(absmax)),
        make_layout(make_shape(n, scale_k, l), stride_S));
  Copy_Scale tiled_copy_scale = {Copy_Scale{}.with(mScale)};

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


  dim3 const block = GemmKernel::get_block_shape();
  dim3 const grid = GemmKernel::get_grid_shape(params);

  const syclcompat::dim3 sycl_block(block.x, block.y, block.z); //workgroup_size: 1*2*1*16, 1, 1
  const syclcompat::dim3 sycl_grid(grid.x, grid.y, grid.z);     //workgroup_number (problem_size / tile_size): N/64, M/16, 1
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
#endif  
}

template void gemm_4bit_inference_cutlass_dequant<sycl::ext::oneapi::bfloat16, 16>(
    int m, int n, int k, sycl::ext::oneapi::bfloat16 *A, unsigned char *B,
    sycl::ext::oneapi::bfloat16 *absmax, float *datatype, float *out, int lda,
    int ldb, int ldc, int blocksize, sycl::queue *stream);

