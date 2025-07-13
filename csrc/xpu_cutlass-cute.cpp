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

using DispatchPolicy = MainloopIntelPVC<Stages>; //, KernelPVC /*Schedule*/>;
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<float /*data_type of GEMM output*/, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;
using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<cutlass::epilogue::IntelPVCEpilogue, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
using SharedStorage = FusionCallBacks::SharedStorage;

using ClusterShape = typename DispatchPolicy::ClusterShape;
  
  //static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>,
  //  "Intel PVC does not support specializing the tile scheduler.");
  using TileScheduler_ = PersistentScheduler; //TileScheduler_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using TileScheduler = typename cutlass::gemm::kernel::detail::TileSchedulerSelector< TileScheduler_, ArchTag, WorkgroupTileShape, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  
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
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;

    // MSVC requires the cast to fix a warning-as-error.
  static constexpr int SharedStorageSize = 0;
  int smem_size = SharedStorageSize;
  
  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize; // sub_group size
  using MmaAtomShape = typename TiledMma::AtomShape_MNK;
  //using SubgroupTileShape = typename CollectiveMainloop::SubgroupTileShape;
  
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
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;

  using traits_load_A = Copy_Traits<GmemTiledCopyA, StrideA>;
  using atom_load_A = Copy_Atom<traits_load_A, ElementA>;
  using val_layout_load_A = decltype(make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{})));
  using Copy_A = decltype(make_tiled_copy(atom_load_A{}, Layout<CopyThreadShape>{}, val_layout_load_A{}));

  using traits_load_B = Copy_Traits<GmemTiledCopyB, StrideB>;
  using atom_load_B = Copy_Atom<traits_load_B, ElementB>;
  using val_layout_load_B = decltype(make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{})));
  using Copy_B = decltype(make_tiled_copy(atom_load_B{}, Layout<CopyThreadShape>{}, val_layout_load_B{}));
  
static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
}

#if 0
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
#endif

//template <typename T, size_t GROUP_SIZE, size_t NUM_PER_THREAD,
//          size_t SUBG_SIZE, int BITS>
template <typename T, int BITS>
class kgemv_4bit_inference_cutlass_cute {
public:
  // Kernel level shared memory storage
  struct SharedStorage {
    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
    EpilogueTensorStorage epilogue;
  };

  struct Params {
    int m, n, k;
    T *A, *B;
    float *absmax, *out;
    const float *datatype;
    int lda, ldb, ldc;
    int blocksize;
	  
	  GemmUniversalMode mode{};
    ProblemShape problem_shape{};
	
    //inloopParams mainloop{};
	  Copy_A tiled_copy_a;
    Copy_B tiled_copy_b;
	
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

#if 0
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
	
    //MainloopArguments mainloop{};
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  
    EpilogueArguments epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
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
#endif 
  static dim3
  get_grid_shape(Params const& params) {
    dim3 grid = TileScheduler::get_tiled_cta_shape_mnl(params.problem_shape, TileShape{}, ClusterShape{});
    if(params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN) {
      return {grid.y, grid.x, grid.z};
    } else {
      return {grid.x, grid.y, grid.z};
    }
  }
  //static
  //cutlass::Status
  //initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr, 
  //  CudaHostAdapter* cuda_adapter = nullptr) {
  //  return Status::kSuccess;
  //}

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
	  /*(int M, int N, int K, T *A, T *B,
                             float *absmax, const float *datatype, float *out,
                             int lda, int ldb, int ldc, int blocksize) {*/
    int M = params.m;
    int N = params.n;
    int K = params.k;
    T* A = params.A;
    T* B = params.B;
    float* absmax = params.absmax;
    float* out = params.out;
    float* datatype = params.datatype;
    int lda = params.lda;
    int ldb = params.ldb;
    int ldc = params.ldc;
    int blocksize = params.blocksize;
    auto tiled_copy_a = params.tiled_copy_a;
    auto tiled_copy_b = params.tiled_copy_b;
   
    int L = 1;
  
    auto problem_size = ProblemShape{M, N, K, L};
       
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
  	
    // Get the appropriate blocks for this sub_group -- potential for sub_group locality
    int thread_idx = int(ThreadIdxX());
    auto blk_shape = TileShape{};
    int m_coord, n_coord, l_coord;
    if (params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN) {
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
         
//Run MainLoop	  
    auto thr_copy_A = tiled_copy_a.get_slice(thread_idx);
    auto thr_copy_B = tiled_copy_b.get_slice(thread_idx);
  
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
    CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
    auto problem_shape_MNKL = append<4>(params.problem_shape, 1);
    epilogue(
      problem_shape_MNKL,
      subgroup_shape, // TODO(codeplay): Inconsistency here w/ blk_coord_mnkl
      blk_coord_mnkl,
      accumulators,
      tiled_mma,
      thread_idx
    );
  }    
};

template <typename T, int BITS>
void gemv_4bit_inference_cutlass_cute(int m, int n, int k, T *A, T *B,
                         float *absmax, float *datatype, float *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream) {

  //const size_t GROUP_SIZE = 128; // workgroup_size
  //const size_t SUBG_SIZE = 32;   // subgroup_size
  //const size_t NUM_PER_THREAD = GROUP_SIZE / SUBG_SIZE;
  //size_t workgroup_num = (n + NUM_PER_THREAD - 1) / NUM_PER_THREAD;
std::cout<<"this is gemv_4bit_inference_cutlass_cute !!!!!!\n";
  auto problem_size = ProblemShape{m, n, k, 1};

  using GemmKernel = kgemv_4bit_inference_cutlass_cute<T, BITS>; //, GROUP_SIZE, NUM_PER_THREAD, SUBG_SIZE, BITS>;

  //dim3 const block = get_block_shape();
  //dim3 const grid = get_grid_shape(params);
  //dim3 grid = get_tiled_cta_shape_mnl(problem_shape); //, TileShape{}); //, ClusterShape{});


  //const syclcompat::dim3 sycl_block(block.x, block.y, block.z);
  //const syclcompat::dim3 sycl_grid(grid.x, grid.y, grid.z);

  // configure smem size and carveout
  //const int smem_size = 0; //GemmKernel::SharedStorageSize;
  static constexpr int smem_size= 0;

  //printf("Host Grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
  //printf("Host Block: (%d, %d, %d)\n", block.x, block.y, block.z);

  sycl::queue q = *stream; //stream ? *stream : syclcompat::get_default_queue();
  
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  auto problem_shape_MNKL = append<4>(problem_size, 1);
  float alpha=1.0;
  float beta=0.f;
  int l = 1;
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));	
//Init Params 
  using Params = GemmKernel::Params;
  Params params;
  params.m = m;
  params.n = n;
  params.k = k;
  params.A = A;
  params.B = B;
  params.absmax = absmax;
  params.out = out;
  params.datatype = datatype;
  params.lda = lda;
  params.ldb = ldb;
  params.ldc = ldc;
  params.blocksize = blocksize;
  
  params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  params.problem_shape = problem_size;
  params.epilogue = CollectiveEpilogue::to_underlying_arguments(problem_size, {{alpha, beta}, nullptr, stride_C, out, stride_C}, nullptr);
  params.hw_info = hw_info;
  TileSchedulerArguments scheduler{};
  params.scheduler = TileScheduler::to_underlying_arguments(
      problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, scheduler, nullptr);;
  
  auto mA_mkl = make_tensor(make_gmem_ptr(A), make_layout(make_shape(m, k, l), stride_A));
  auto mB_nkl = make_tensor(make_gmem_ptr(B), make_layout(make_shape(n, k, l), stride_B));
  Copy_A tiled_copy_a{Copy_A{}.with(mA_mkl)};
  Copy_B tiled_copy_b{Copy_B{}.with(mB_nkl)};
  
  params.tiled_copy_a = tiled_copy_a;
  params.tiled_copy_b = tiled_copy_b;
  
  dim3 const block = get_block_shape();
  dim3 const grid = GemmKernel::get_grid_shape(params);
  cutlass::arch::synclog_setup();
  cutlass::kernel_launch<GemmKernel, Params>(grid, block, smem_size, stream, params, false);
  syclcompat::wait();
}

template void gemv_4bit_inference_cutlass_cute<sycl::ext::oneapi::bfloat16, 16>(
    int m, int n, int k, sycl::ext::oneapi::bfloat16 *A, sycl::ext::oneapi::bfloat16 *B,
    float *absmax, float *datatype, float *out, int lda,
    int ldb, int ldc, int blocksize, sycl::queue *stream);

