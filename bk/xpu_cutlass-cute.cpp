#include "xpu_cutlass.h"
#include <xpu_ops.h>
#include <bit>
#include <cmath>
#include <iostream>

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

using ElementA = bfloat16_t;
using ElementB = bfloat16_t;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;      // data_type of accumulator
using ElementComputeEpilogue = float;  // data_type of epilogue operations
using ElementOutput = float;
static constexpr int Stages = 2;

using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;
using CopyOpG2R = XE_2D_U32x8x16_LD_N;
using CopyOpR2G = XE_2D_U32x8x16_ST_N;
using GmemTiledCopyC = CopyOpG2R;
using GmemTiledCopyD = cute::conditional_t<not cute::is_void_v<ElementD> && not cute::is_void_v<CopyOpR2G>,
                                             CopyOpR2G, XE_2D_U32x8x16_ST_N>;

using StrideA = cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>;
using StrideB = cutlass::gemm::TagToStrideB_t<cutlass::layout::RowMajor>;
using StrideC = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;
using StrideD = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;
using ProblemShape = Shape<int, int, int, int>;

using TileShape = Shape<_256, _256, _32>;
using WorkgroupTileShape = TileShape;
using TiledMma =
    typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
                                  Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});
using DispatchPolicy = MainloopIntelPVC<Stages, KernelPVC /*Schedule*/>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<float /*data_type of GEMM output*/, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<cutlass::epilogue::IntelPVCEpilogue, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

//  struct TensorStorageImpl: cute::tuple<SmemCStorage, SmemDStorage> {
//    using FusionStorage = typename FusionCallbacks::SharedStorage;
//    FusionStorage thread;
//  };
//
//  struct SharedStorage {
//    using TensorStorage = TensorStorageImpl;
//
//    TensorStorage tensors;
//  };
//  using TensorStorage = typename SharedStorage::TensorStorage;
//
//  // Kernel level shared memory storage
//  struct SharedStorage {
//    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
//    EpilogueTensorStorage epilogue;
//  };
  using SharedStorage = FusionCallBacks::SharedStorage;
static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
}

static dim3
get_tiled_cta_shape_mnl(ProblemShape problem_shape) {
  using cta_shape = TileShape;
  auto cta_m = (get<0>(problem_shape) + get<0>(cta_shape{}) - 1) / get<0>(cta_shape{});
  auto cta_n = (get<1>(problem_shape) + get<1>(cta_shape{}) - 1) / get<1>(cta_shape{});

  return {
    static_cast<uint32_t>(cta_m),
    static_cast<uint32_t>(cta_n),
    static_cast<uint32_t>(get<3>(problem_shape))
  };
}

template <typename T, size_t GROUP_SIZE, size_t NUM_PER_THREAD,
          size_t SUBG_SIZE, int BITS>
class kgemv_4bit_inference_cutlass {
public:
  struct Params {
      int m, n, k;
      T *A, *B;
      float *absmax, *out;
      const float *datatype;
      int lda, ldb, ldc;
      int blocksize;
  };

  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    //MainloopArguments mainloop{};
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  
    //EpilogueArguments epilogue{};
    //typename FusionCallbacks::Arguments thread{};
    ElementC const* ptr_C;
    StrideC dC;
    ElementD* ptr_D;
    StrideD dD;
  
    //cutlass::KernelHardwareInfo hw_info{};
    //TileSchedulerArguments scheduler{};
  };
  
  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    if (args.mode == GemmUniversalMode::kGemmSplitKParallel) {
      workspace_bytes += sizeof(int) * size_t(cute::size<0>(TileShape{})) * size_t(cute::size<1>(TileShape{}));
    }
  
    //TODO: Check it!!
    workspace_bytes += 0; //GemmKernel::get_workspace_size(args);
  
    CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);
  
    return workspace_bytes;
  }

  //static
  //cutlass::Status
  //initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr, 
  //  CudaHostAdapter* cuda_adapter = nullptr) {
  //  return Status::kSuccess;
  //}

#if 0
  kgemv_4bit_inference_cutlass(int M_, int N_, int K_, T *A_, T *B_,
                             float *absmax_, const float *datatype_, float *out_,
                             int lda_, int ldb_, int ldc_, int blocksize_)
      : M(M_), N(N_), K(K_), A(A_), B(B_),
        absmax(absmax_), out(out_), datatype(datatype_),
        lda(lda_), ldb(ldb_), ldc(ldc_), blocksize(blocksize_) {}

private:
  int M;
  int N;
  int K;
  T *A;
  T *B;
  float *absmax;
  const float *datatype;
  float *out;
  int lda;
  int ldb;
  int ldc;
  int blocksize;
  int SharedStorageSize = 0;

public:
  CUTLASS_DEVICE
  void operator()(sycl::nd_item<1> item) const {
#endif
#if 1
  CUTLASS_DEVICE
  void operator()(int M, int N, int K, T *A, T *B,
                             float *absmax, const float *datatype, float *out,
                             int lda, int ldb, int ldc, int blocksize) {//(sycl::nd_item<1> item) const {
#else
  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) const {                              
    //SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    auto M = params.m;
    auto N = params.n;
    auto K = params.k;
    auto A = params.A;
    auto B = params.B;
    auto out = params.out;
    auto absmax = params.absmax;
    auto datatype = params.datatype;
    auto lda = params.lda;
    auto ldb = params.ldb;
    auto ldc = params.ldc;
    auto blocksize = params.blocksize;
#endif
#if 1    
    int L = 1;
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    //sycl::ext::oneapi::experimental::printf("this is kgemv_4bit_inference_cutlass ...\n");
    //sycl::stream(1024, 128) << "this is kgemv_4bit_inference_cutlass ...\n" << sycl::endl;
  
    //cutlass::KernelHardwareInfo hw_info;
    //hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  
    auto problem_size = ProblemShape{M, N, K, L};
    static constexpr int SharedStorageSize = 0;
    
    //float alpha=1.0;
    //float beta=0.f;
    
    Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,    
      (bfloat16_t*)A, stride_A, (bfloat16_t*)B, stride_B,
      /*{alpha, beta},*/ nullptr, stride_C, out, stride_C}; //, hw_info};
  
    //size_t workspace_size = 0; //get_workspace_size(arguments);
    //cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  
    //CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    
    //Create TiledMMA
    // Workgroup-level tile
    using TileShape = Shape<_256, _256, _32>;
    using WorkgroupTileShape = TileShape;
    using TiledMma = 
        typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
                                      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
//Run Adapter
      /*using TileSchedulerTag = TileScheduler_;
      using TileScheduler = typename detail::TileSchedulerSelector<
      TileScheduler_, ArchTag, WorkgroupTileShape,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
  	using TileSchedulerArguments = typename TileScheduler::Arguments;
      using TileSchedulerParams = typename TileScheduler::Params;*/
  	
    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
  	
    // Get the appropriate blocks for this sub_group -- potential for sub_group locality
    int thread_idx = int(ThreadIdxX());
    auto blk_shape = TileShape{};
    int m_coord, n_coord, l_coord;
    if (0) { //params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN) {
      m_coord = BlockIdxY();
      n_coord = BlockIdxX();
      l_coord = BlockIdxZ();
    } else {
      m_coord = BlockIdxX();
      n_coord = BlockIdxY();
      l_coord = BlockIdxZ();
    }
  
    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
    constexpr auto workgroup_shape = WorkgroupTileShape{}; 
  
    //(SUB_M,SUB_N,SUB_K)
    static constexpr auto BLK_M = get<0>(WorkgroupTileShape{});
    static constexpr auto BLK_N = get<1>(WorkgroupTileShape{});
    static constexpr auto BLK_K = get<2>(WorkgroupTileShape{});
  
    static constexpr auto ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());
  
    static_assert(BLK_M % TiledMma{}.template tile_size_mnk<0>() == 0, "TiledMma permutation size must match block size.");
    static_assert(BLK_N % TiledMma{}.template tile_size_mnk<1>() == 0, "TiledMma permutation size must match block size.");
    static_assert(BLK_K % TiledMma{}.template tile_size_mnk<2>() == 0, "TiledMma permutation size must match block size.");
  
    static constexpr auto SG_M = ceil_div(BLK_M, ATOM_M);
    static constexpr auto SG_N = ceil_div(BLK_N, ATOM_N);
    static constexpr auto SG_K = ceil_div(BLK_K, ATOM_K);
    using SubgroupTileShape = Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;
    
      // 32
    static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;
    //static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});
  
    using CopyThreadShape = Shape<_1, Int<DispatchPolicy::SubgroupSize>>;
  
    using traits_load_A = Copy_Traits<GmemTiledCopyA, StrideA>;
    using atom_load_A = Copy_Atom<traits_load_A, ElementA>;
    using val_layout_load_A = decltype(make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{})));
    using Copy_A = decltype(make_tiled_copy(atom_load_A{}, Layout<CopyThreadShape>{}, val_layout_load_A{}));
  
    using traits_load_B = Copy_Traits<GmemTiledCopyB, StrideB>;
    using atom_load_B = Copy_Atom<traits_load_B, ElementB>;
    using val_layout_load_B = decltype(make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{})));
    using Copy_B = decltype(make_tiled_copy(atom_load_B{}, Layout<CopyThreadShape>{}, val_layout_load_B{}));
    
    
    constexpr auto subgroup_shape = SubgroupTileShape{};                   
  
    Tensor mA_mkl = cute::get_pvc_tensor(make_shape(M,K,L));   //(m,k,l)
    Tensor mB_nkl = cute::get_pvc_tensor(make_shape(N,K,L));   //(n,k,l)
  
    Tensor gA = local_tile(mA_mkl, select<0,2>(blk_shape), make_coord(m_coord,_,l_coord));
    Tensor gB = local_tile(mB_nkl, select<1,2>(blk_shape), make_coord(n_coord,_,l_coord));	
  
    // Allocate the tiled_mma and the accumulators for the (M,N) subgroup_shape
    TiledMma tiled_mma;
  
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); 
    clear(accumulators);
  
    auto k_tile_iter  = cute::make_coord_iterator(idx2crd(0, make_shape(K)), make_shape(K));
    int  k_tile_count = ceil_div(K, get<2>(workgroup_shape));
  
//Initialize
    // Initialize the workspace
    //initialize_workspace(arguments, workspace, nullptr, nullptr);
  
    // Initialize the Params structure
    //params_ = GemmKernel::to_underlying_arguments(args, workspace);
  
    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    //  static
    //  Params
    //  to_underlying_arguments(Arguments const& args, void* workspace) {
    //(void) workspace;
    auto problem_shape_MNKL = append<4>(arguments.problem_shape, 1);
  
    //auto mainloop_args = CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace);
  	//(void) workspace;
  
    //auto [M,N,K,L] = arguments.problem_shape;
  
    auto mA_mkl_cp = make_tensor(make_gmem_ptr(arguments.ptr_A), make_layout(make_shape(M, K, L), arguments.dA));
    auto mB_nkl_cp = make_tensor(make_gmem_ptr(arguments.ptr_B), make_layout(make_shape(N, K, L), arguments.dB));
    Copy_A tiled_copy_a{Copy_A{}.with(mA_mkl_cp)};
    Copy_B tiled_copy_b{Copy_B{}.with(mB_nkl_cp)};
    
    //auto mainloop_args = {tiled_copy_a, tiled_copy_b};
  
    //TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
    //  problem_shape_MNKL, TileShape{}, ClusterShape{}, args.hw_info, args.scheduler, &workspace);
    /*return {
      arguments.mode,
      arguments.problem_shape,
      mainloop_args,
      CollectiveEpilogue::to_underlying_arguments(arguments.problem_shape, arguments.epilogue, workspace),
      arguments.hw_info,
      scheduler
    };*/
  
  
    using CopyThreadShape = Shape<_1, Int<cutlass::epilogue::IntelPVCEpilogue::SubgroupSize>>;
    
    using Trait_C = Copy_Traits<GmemTiledCopyC, StrideC>;
    using val_layout_load_C = decltype(make_layout(shape_div(typename Trait_C::BlockShape{}, CopyThreadShape{})));
    using XE_Copy_C = decltype(make_tiled_copy(Copy_Atom<Trait_C, ElementC>{}, Layout<CopyThreadShape>{}, val_layout_load_C{}));
  
    using Trait_D = Copy_Traits<GmemTiledCopyD, StrideD>;
    using val_layout_store_D = decltype(make_layout(shape_div(typename Trait_D::BlockShape{}, CopyThreadShape{})));
    using XE_Copy_D = decltype(make_tiled_copy(Copy_Atom<Trait_D, ElementD>{}, Layout<CopyThreadShape>{}, val_layout_store_D{}));
    
    constexpr static bool is_source_supported = not cute::is_void_v<ElementC>;
    constexpr static bool is_destination_supported = not cute::is_void_v<ElementD> && not cute::is_void_v<CopyOpR2G>;
  
    //constexpr static bool is_m_major_C = detail::is_m_major<StrideC>();
    //constexpr static bool is_m_major_D = detail::is_m_major<StrideD>();
    
    XE_Copy_C xe_load_c = {};
    if constexpr (is_source_supported) {
      auto mC = make_tensor(make_gmem_ptr(arguments.ptr_C), make_layout(make_shape(M, N, L), arguments.dC));
      xe_load_c = {xe_load_c.with(mC)};
    }
  
    XE_Copy_D xe_store_d = {};
    if (is_destination_supported) {
      auto mD = make_tensor(make_gmem_ptr(arguments.ptr_D), make_layout(make_shape(M, N, L), arguments.dD));
      xe_store_d = {xe_store_d.with(mD)};
    }
  
    /*return {
      FusionCallbacks::to_underlying_arguments(problem_shape, arguments.thread, workspace),
      xe_load_c,
      xe_store_d,
    }*/;
  
  
    int smem_size = SharedStorageSize;
    
//Run MainLoop	
    //(void)blk_coord;
    //static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    //static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
  
    auto thr_copy_A = tiled_copy_a.get_slice(thread_idx);
    auto thr_copy_B = tiled_copy_b.get_slice(thread_idx);
  
    // Instantiate the MMA object and get thread slice
    //TiledMma tiled_mma;
    // TODO(Codeplay): see if we can make this nicer
    // To make all work items in a subgroup have the same global tensors pass in the index of work item 0 in each subgroup
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * DispatchPolicy::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);
  
    // Partition global counting tensors for MMA
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);
  
    Tensor tCrA = make_tensor<ElementA>(make_fragment_layout(tiled_copy_a, tCgA(_,_,_,0).shape()));
    Tensor tCrB = make_tensor<ElementB>(make_fragment_layout(tiled_copy_b, tCgB(_,_,_,0).shape()));
  
    // Retile registers for copies
    Tensor tArA = thr_copy_A.retile_D(tCrA);
    Tensor tBrB = thr_copy_B.retile_D(tCrB);
    
    // Retile global counting tensors for copies
    Tensor tAgA = thr_copy_A.retile_S(tCgA);
    Tensor tBgB = thr_copy_B.retile_S(tCgB);
    
    auto tiled_prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>,Int<BLK_K>>, Num_SGs>(tiled_copy_a);
    auto tiled_prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>,Int<BLK_K>>, Num_SGs>(tiled_copy_b);
    auto thr_prefetch_A = tiled_prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = tiled_prefetch_b.get_slice(thread_idx);
    
    // Partition global tile for prefetch
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);
  
    //
    // Mainloop
    //
    const auto k_start_idx = crd2idx((*k_tile_iter), make_shape(K));
    constexpr int barrier_scope = 2;
    int prefetch_k = k_start_idx;
  
    CUTLASS_PRAGMA_UNROLL
    for (; prefetch_k < DispatchPolicy::Stages; prefetch_k++) {
      prefetch(tiled_prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(tiled_prefetch_b, pBgB(_, _, _, prefetch_k));
    }
  
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = k_start_idx; k_tile < k_tile_count + k_start_idx; k_tile++, prefetch_k++) {
      barrier_arrive(barrier_scope);
      // Copy gmem to rmem for the first k_tile
      copy(tiled_copy_a, tAgA(_,_,_,k_tile), tArA);
      copy(tiled_copy_b, tBgB(_,_,_,k_tile), tBrB);
  
      if (prefetch_k < k_tile_count) {
        prefetch(tiled_prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(tiled_prefetch_b, pBgB(_, _, _, prefetch_k));
      }
  
      cute::gemm(tiled_mma, tCrA, tCrB, accumulators);
      barrier_wait(barrier_scope);
    }
    
//Run Epilogue
      //(void) tiled_mma;
  
    using DispatchPolicy_Epi = cutlass::epilogue::IntelPVCEpilogue;
    static constexpr int SubgroupSize = DispatchPolicy_Epi::SubgroupSize;
    using CtaTileMNK = TileShape;
    static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");
  
    using MmaAtomShape = typename TiledMma::AtomShape_MNK;
    static constexpr auto BLK_M_EPI = get<0>(CtaTileMNK{});
    static constexpr auto BLK_N_EPI = get<1>(CtaTileMNK{});
    static constexpr auto BLK_K_EPI = get<2>(CtaTileMNK{});
    // static_assert(is_same_v<typename TiledMma::ThrLayoutVMNK, int>, "assertation fail");
    static constexpr auto ATOM_M_EPI = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N_EPI = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_K_EPI = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());
    
  //  static_assert(
  //    BLK_M_EPI = ATOM_M_EPI % 0 &&
  //    BLK_N_EPI = ATOM_N_EPI % 0 &&
  //    BLK_K_EPI = ATOM_K_EPI % 0,
  //    "expected CTATileMNK to be evenly divided by TiledMma::ThrLayoutVMNK");
    static constexpr auto SG_M_EPI = BLK_M_EPI / ATOM_M_EPI;
    static constexpr auto SG_N_EPI = BLK_N_EPI / ATOM_N_EPI;
    static constexpr auto SG_K_EPI = BLK_K_EPI / ATOM_K_EPI;
    using SubgroupTileShape = Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;
  
    static constexpr int FragsM = get<0>(SubgroupTileShape{}) / get<0>(MmaAtomShape()); // A frags per sub_group
    static constexpr int FragsN = get<1>(SubgroupTileShape{}) / get<1>(MmaAtomShape()); // B frags per sub_group
  
    static constexpr int FragmentSize = (get<0>(MmaAtomShape()) * get<1>(MmaAtomShape())) / SubgroupSize;
  
    // Indexing variables
  	auto problem_shape_mnkl = cute::append<4>(problem_size, 1);
    //auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord_epi, n_coord_epi, k_coord_epi, l_coord_epi] = blk_coord_mnkl;
    auto m_sg = get_sub_group_id() / ATOM_N_EPI;
    auto n_sg = get_sub_group_id() % ATOM_N_EPI;
  
    using EpilogueTile = decltype(get<0>(xe_store_d.get_layoutS_MN()).shape());
  
    auto sg_local_m_coord = get_sub_group_id() / ATOM_N_EPI;
    auto sg_local_n_coord = get_sub_group_id() % ATOM_N_EPI;
  
    auto sg_m_coord = m_coord_epi * ATOM_M_EPI + sg_local_m_coord;
    auto sg_n_coord = n_coord_epi * ATOM_N_EPI + sg_local_n_coord;
    auto sg_coord = make_coord(sg_m_coord, sg_n_coord, k_coord_epi, l_coord_epi);
  
    FusionCallBacks fusion_callbacks;
    bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();
    
    // Represent the full output tensor
    Tensor mD_mnl = cute::get_pvc_tensor(make_shape(M,N,L));
  
    // Tile the output tensor per WG and select the tile for current WG
    Tensor g_wg_D = local_tile(mD_mnl, take<0,2>(CtaTileMNK{}), make_coord(m_coord_epi,n_coord_epi,l_coord_epi));  // (BLK_M,BLK_N)
    
    // Tile the output tensor per SG and select tile for the current SG
    Tensor gD = local_tile(g_wg_D, take<0,2>(SubgroupTileShape{}), make_coord(m_sg,n_sg));            // (SG_M,SG_N)
  
    auto thread_xe_store_d = xe_store_d.get_thread_slice(thread_idx);
    Tensor tCgD = thread_xe_store_d.partition_D(gD);
  
    Tensor trC = make_tensor<typename TiledMma::ValTypeC>(Shape<Int<FragmentSize>>{});
    Tensor trD = make_tensor<typename TiledMma::ValTypeD>(Shape<Int<FragmentSize>>{});
  
    // Because Sm90 uses shared memory, they are not tied to using the same accumulator values
    // for MMA and Epilogue. But because we are operating directly in the accumulators, we need to be
    // sure that we are operating on the same values.
    ThrCopy thread_g2r = xe_load_c.get_slice(thread_idx);
  
    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    Tensor cD = local_tile(mD_crd, take<0,2>(SubgroupTileShape{}), make_coord(sg_m_coord, sg_n_coord));
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(CtaTileMNK{}), make_coord(m_coord_epi, n_coord_epi));          // (CTA_M,CTA_N)
    Tensor tRS_cD_mn = thread_g2r.partition_S(flat_divide(cD_mn, EpilogueTile{}));     // (G2R,G2R_M,G2R_N,EPI_M,EPI_N)
  
    Tensor tRS_cD = make_counting_tensor(tRS_cD_mn.layout());                          // (G2R,G2R_M,G2R_N,EPI_M,EPI_N)
  
    // Get the fusion callbacks
    // Arguments passed here relate to sub-group tiles, rather than CTA (work-group) tiles
    constexpr bool RefSrc = true;
    auto residue_mn = make_coord(M, N); //TODO(Codeplay): this is not correct
    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
                      problem_shape_mnkl,
                      SubgroupTileShape{},
                      sg_coord,
                      tiled_mma,
                      EpilogueTile{},
                      xe_store_d,
                      cD,
                      residue_mn,
                      tRS_cD,
                      residue_mn,
                      trC,
                      thread_idx,
                    };
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<RefSrc>(cst_args);
  
    cst_callbacks.begin();
  
    auto acc_frag = recast<Array<ElementOutput, FragmentSize>>(accumulators);
    auto trD_frag = recast<Array<ElementOutput, FragmentSize>>(trD);
  
    constexpr int ValuesLoaded =
      FragsM * FragsN * FragmentSize * SubgroupSize * ATOM_M * ATOM_N * ATOM_K;
    constexpr int MN = get<0>(CtaTileMNK{}) * get<1>(CtaTileMNK{});
    static_assert(ValuesLoaded == MN, "the total elements loaded by all threads should be the same as MxN" );
    
    auto synchronize = [&] () {};
    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < FragsN; epi_n++) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < FragsM; epi_m++) {
        cst_callbacks.begin_loop(epi_m, epi_n);
  
        if (is_C_load_needed) {
          //cordinates for C and D are the same
          copy(xe_load_c, tCgD(_, epi_m, epi_n), trC);
        }
  
        cst_callbacks.previsit(epi_m, epi_n, 0, is_C_load_needed);
  
        auto acc_frag_mn = acc_frag(_, epi_m, epi_n);
  
        CUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < size<0>(trD_frag); ++epi_v) {
          trD_frag(epi_v) = cst_callbacks.visit(acc_frag_mn(epi_v), epi_v, epi_m, epi_n);
        }
        cst_callbacks.reduce(nullptr, synchronize, epi_m, epi_n, (epi_m == FragsM - 1 && epi_n == FragsN - 1), trD_frag);
        
        if constexpr (is_destination_supported) {
          copy(xe_store_d, trD, tCgD(_, epi_m, epi_n));
        }
        
        cst_callbacks.end_loop(epi_m, epi_n);
      }
    }
  
    cst_callbacks.end();
#endif 
  }
};

//template class kgemv_4bit_inference_cutlass<sycl::ext::oneapi::bfloat16, 128, 4, 32, 16>;

template <typename T, int BITS>
void gemv_4bit_inference_cutlass(int m, int n, int k, T *A, T *B,
                         float *absmax, float *datatype, float *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream) {

  //auto &queue = *stream;
  
  const size_t GROUP_SIZE = 128; // workgroup_size
  const size_t SUBG_SIZE = 32;   // subgroup_size
  const size_t NUM_PER_THREAD = GROUP_SIZE / SUBG_SIZE;
  size_t workgroup_num = (n + NUM_PER_THREAD - 1) / NUM_PER_THREAD;

  auto problem_shape = ProblemShape{m, n, k, 1};

#if 0  
  dim3 const block = get_block_shape();
  //dim3 const grid = get_grid_shape(params);
  dim3 grid = get_tiled_cta_shape_mnl(problem_shape); //, TileShape{}); //, ClusterShape{});


  const syclcompat::dim3 sycl_block(block.x, block.y, block.z);
  const syclcompat::dim3 sycl_grid(grid.x, grid.y, grid.z);

  auto &queue = *stream;
  kgemv_4bit_inference_cutlass<T, GROUP_SIZE, NUM_PER_THREAD, SUBG_SIZE, BITS> kfn(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);
    sycl_kernel_submit<decltype(kfn), 1, 32>(
        sycl::nd_range<1>(sycl::range<1>(GROUP_SIZE * workgroup_num),
                          sycl::range<1>(GROUP_SIZE)),
        queue, kfn);
  queue.wait();
#else  
  using GemmKernel = kgemv_4bit_inference_cutlass<T, GROUP_SIZE, NUM_PER_THREAD, SUBG_SIZE, BITS>;//(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);
  using GemmKernel_t = GetUnderlyingKernel_t<GemmKernel>;

  dim3 const block = get_block_shape();
  //dim3 const grid = get_grid_shape(params);
  dim3 grid = get_tiled_cta_shape_mnl(problem_shape); //, TileShape{}); //, ClusterShape{});


  const syclcompat::dim3 sycl_block(block.x, block.y, block.z);
  const syclcompat::dim3 sycl_grid(grid.x, grid.y, grid.z);

  // configure smem size and carveout
  //const int smem_size = 0; //GemmKernel::SharedStorageSize;
  static constexpr int smem_size= 0;

  //Status launch_result{ Status::kSuccess };
  //  launch_result = Status::kSuccess;
  //cutlass::arch::synclog_setup();

  sycl::queue q = *stream; //stream ? *stream : syclcompat::get_default_queue();

  using Params = GemmKernel_t::Params;
#if 1
  cutlass::kernel_launch<GemmKernel, Params>(
          grid, block, smem_size, stream, Params{m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize}, false);
  syclcompat::wait();
#else
  using namespace syclcompat::experimental;

//  Params params{
//      .M = m, .N = n, .K = k,
//      .A = A, .B = B,
//      .out = out,
//      .lda = lda, .ldb = ldb, .ldc = ldc
//  };
Params params;
params.m = m;
params.n = n;
params.k = k;
params.A = A;
params.B = B;
params.out = out;
params.lda = lda;
params.ldb = ldb;
params.ldc = ldc;
params.absmax = absmax;
params.datatype = datatype;
params.blocksize = blocksize;
std::cout<<"before run kernel......\n";
  auto event = launch<device_kernel<GemmKernel_t>>(launch_policy{
    sycl_grid, sycl_block//, local_mem_size{static_cast<std::size_t>(smem_size)}
    , kernel_properties{sycl_exp::sub_group_size<DispatchPolicy::SubgroupSize>}
  }, q, params);
// 计算执行范围
//size_t local_size = 256;
//size_t global_size = (m + local_size - 1) / local_size * local_size;
//
//// 启动内核
//auto event = syclcompat::experimental::launch<
//    GemmKernel>(
//    launch_policy{
//        sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},//sycl::nd_range<1>(global_size, local_size),
//        kernel_properties{sycl_exp::sub_group_size<DispatchPolicy::SubgroupSize>}
//    },
//    q,
//    params
//);
  EventManager::getInstance().addEvent(event);
  syclcompat::wait();
#endif
#endif
}

//template class kgemv_4bit_inference_cutlass<sycl::ext::oneapi::bfloat16, 128, 4, 32, 16>;
template void gemv_4bit_inference_cutlass<sycl::ext::oneapi::bfloat16, 16>(
    int m, int n, int k, sycl::ext::oneapi::bfloat16 *A, sycl::ext::oneapi::bfloat16 *B,
    float *absmax, float *datatype, float *out, int lda,
    int ldb, int ldc, int blocksize, sycl::queue *stream);

