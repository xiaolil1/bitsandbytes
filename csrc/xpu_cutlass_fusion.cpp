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

using ElementA = MmaType;
using ElementB = QuantType;

using ElementMMA = ElementA;
using ElementQuant = QuantType;
using ElementScale = float;

using ElementAccumulator = float;
using ElementComputeEpilogue = float;
using ElementOutput = float;

using ProblemShape = Shape<int, int, int, int>;

#if 0
static constexpr float quant_map_static[16] = {
    -1.0f, -0.6961928f, -0.52507305f, -0.39491749f,
    -0.28444138f, -0.18477343f, -0.09105004f, 0.0f,
    0.0795803f, 0.1609302f, 0.2461123f, 0.33791524f,
    0.44070983f, 0.562617f, 0.72295684f, 1.0f
};
#endif 

using TileShape = Shape<_32, _128, _64>;
using TiledMma =
    typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
                                  Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>>::TiledMMA;
using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
constexpr int PipelineStages = 4;

using MmaAtomShape = typename TiledMma::AtomShape_MNK;
using WorkgroupTileShape = TileShape;
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

static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;
static constexpr auto SG_QNT_WIDTH = Int<SG_N>{};
static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

using DispatchPolicy = cutlass::gemm::MainloopIntelPVCMixedPrecision<PipelineStages>;
static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

// Design Scheduler 
using TileScheduler_ = PersistentScheduler;
static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>, "Intel PVC does not support specializing the tile scheduler.");
using ArchTag = typename DispatchPolicy::ArchTag;
using TileScheduler = typename cutlass::gemm::kernel::detail::TileSchedulerSelector<TileScheduler_, ArchTag, WorkgroupTileShape, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
using TileSchedulerArguments = typename TileScheduler::Arguments;
using TileSchedulerParams = typename TileScheduler::Params;

using ClusterShape = typename DispatchPolicy::ClusterShape;

// Define Copy
using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;
using CopyThreadShapeRev = decltype(cute::reverse(CopyThreadShape{}));

using StrideA = cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>;
using traits_load_A = Copy_Traits<GmemTiledCopyA, StrideA>;
using atom_load_A = Copy_Atom<traits_load_A, ElementA>;
using val_layout_load_A = decltype(make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{})));
using Copy_A = decltype(make_tiled_copy(atom_load_A{}, Layout<CopyThreadShape>{}, val_layout_load_A{}));

using GmemTiledCopyB = XE_2D_U4x32x16_LD_T; 
using StrideB = cutlass::gemm::TagToStrideB_t<cutlass::layout::ColumnMajor>;
using traits_load_B = Copy_Traits<GmemTiledCopyB, StrideB>;
using atom_load_B = Copy_Atom<traits_load_B, ElementB>;
using val_layout_load_B = decltype(make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{})));
using Copy_B = decltype(make_tiled_copy(atom_load_B{}, Layout<CopyThreadShape>{}, val_layout_load_B{}));

using GmemTiledCopyScale = XE_2D_U32x1x16_LD_N; 
using StrideScale = cute::Stride<_1, int64_t, int64_t>;
using traits_load_scale = Copy_Traits<GmemTiledCopyScale, StrideScale>;
using atom_load_scale = Copy_Atom<traits_load_scale, ElementScale>;
using val_layout_load_scale = decltype(make_layout(shape_div(typename traits_load_scale::BlockShape{}, CopyThreadShapeRev{}))); 
using Copy_Scale = decltype(make_tiled_copy(atom_load_scale{}, Layout<CopyThreadShapeRev>{}, val_layout_load_scale{}));

using GmemTiledCopyD = XE_2D_U32x8x16_ST_N;
using StrideD = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;
using Trait_D = Copy_Traits<GmemTiledCopyD, StrideD>;
using val_layout_store_D = decltype(make_layout(shape_div(typename Trait_D::BlockShape{}, CopyThreadShape{})));
using Copy_D = decltype(make_tiled_copy(Copy_Atom<Trait_D, ElementOutput>{}, Layout<CopyThreadShape>{}, val_layout_store_D{}));

template <typename T, int BITS>
class gemm_4bit_cutlass_kernel {
public:
  struct Params {
    int m, n, k, l;
    T* A;
    uint8_t* B;
    float* out;
    float *datatype; //LUT
    int group_size;
    float* absmax;
	  
    ProblemShape problem_shape{};

  	Copy_A tiled_copy_a;
    Copy_B tiled_copy_b;
	  Copy_Scale tiled_copy_scale;
    Copy_D tiled_store_d;
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

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    int thread_idx = int(ThreadIdxX());

    // Load Dequatize LUT and save to SLM, 16 for 4bits
    float* quant_map = reinterpret_cast<float*>(smem_buf);
    if (thread_idx < 16) {
      quant_map[thread_idx] = params.datatype[thread_idx];
    }
    barrier_arrive(3);

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
  
    Tensor mA_mkl = cute::get_pvc_tensor(make_shape(params.m, params.k, params.l));
    Tensor mB_nkl = cute::get_pvc_tensor(make_shape(params.n, params.k,1));
  
    Tensor gA = local_tile(mA_mkl, select<0,2>(TileShape{}), make_coord(m_coord,_,l_coord));
    Tensor gB = local_tile(mB_nkl, select<1,2>(TileShape{}), make_coord(n_coord,_,0));		
  
    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(TileShape{})); 
    clear(accumulators);

    auto k_tile_iter  = cute::make_coord_iterator(idx2crd(0, make_shape(params.k)), make_shape(params.k));
    int k_tile_count = ceil_div(params.k, get<2>(WorkgroupTileShape{}));

    auto thr_copy_A = params.tiled_copy_a.get_slice(thread_idx);
    auto thr_copy_B = params.tiled_copy_b.get_slice(thread_idx);
	  auto thr_copy_scale = params.tiled_copy_scale.get_slice(thread_idx);
  
    auto first_thread_in_sg_idx = syclcompat::get_nd_item<1>().get_sub_group().get_group_linear_id() * DispatchPolicy::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx); 
  
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);
	
    Tensor mma_A = make_tensor<ElementMMA>(make_fragment_layout(params.tiled_copy_a, tCgA(_,_,_,0).shape()));
    Tensor mma_B = make_tensor<ElementMMA>(make_fragment_layout(params.tiled_copy_b, tCgB(_,_,_,0).shape()));

	  Tensor dequant_frag = make_tensor<ElementB>(mma_B.layout());

    static constexpr auto scale_traits_size = decltype(size(typename GmemTiledCopyScale::BlockShape{}))::value / DispatchPolicy::SubgroupSize;
    static constexpr auto scale_traits_num = SG_QNT_WIDTH / decltype(size<1>(typename GmemTiledCopyScale::BlockShape{}))::value;
    using FragScaleLayout = Layout<Shape<Int<scale_traits_size>, Int<scale_traits_num>, _1>>;
    Tensor fragment_scale = make_tensor<ElementScale>(FragScaleLayout{});
    
    static_assert(std::is_same_v<typename decltype(dequant_frag)::value_type, ElementQuant>);
    static_assert(std::is_same_v<typename decltype(mma_A)::value_type, ElementMMA>);
    static_assert(std::is_same_v<typename decltype(mma_B)::value_type, ElementMMA>);

    Tensor frag_copy_A = thr_copy_A.retile_D(mma_A);
    Tensor frag_copy_B = thr_copy_B.retile_D(dequant_frag);
    Tensor frag_copy_Scale = thr_copy_scale.retile_D(fragment_scale);

    Tensor tAgA = thr_copy_A.retile_S(tCgA);
    Tensor tBgB = thr_copy_B.retile_S(tCgB);

    auto tiled_prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>,Int<BLK_K>>, Num_SGs>(params.tiled_copy_a);;
    auto tiled_prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>,Int<BLK_K>>, Num_SGs>(params.tiled_copy_b);;
    auto thr_prefetch_A = tiled_prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = tiled_prefetch_b.get_slice(thread_idx);

    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);
	
    const int k_reload_factor = ceil_div(params.group_size, BLK_K);

    auto tSgS = [&](){
        return make_tensor(make_inttuple_iter(make_coord(n_coord * BLK_N + get<2>(thr_mma.thr_vmnk_)*SG_QNT_WIDTH, 0, 0)),
                          make_layout(make_shape(Int<scale_traits_size>{}, Int<scale_traits_num>{}, _1{}, k_tile_count/k_reload_factor),
                                      make_stride(E<0>{}*_16{}, E<0>{}*_16{}, _0{}, E<1>{}*_1{})));
      
    }();

    static constexpr int FragsM = get<0>(SubgroupTileShape{}) / get<0>(MmaAtomShape()); // A frags per sub_group
    static constexpr int FragsN = get<1>(SubgroupTileShape{}) / get<1>(MmaAtomShape()); // B frags per sub_group

    static constexpr int FragmentSize = (get<0>(MmaAtomShape()) * get<1>(MmaAtomShape())) / SubgroupSize;

    auto m_sg = get_sub_group_id() / ATOM_N;
    auto n_sg = get_sub_group_id() % ATOM_N;

    // Represent the full output tensor
    Tensor mD_mnl = cute::get_pvc_tensor(make_shape(params.m, params.n, params.l));

    // Tile the output tensor per WG and select the tile for current WG
    Tensor g_wg_D = local_tile(mD_mnl, take<0,2>(WorkgroupTileShape{}), make_coord(m_coord,n_coord,l_coord));  // (BLK_M,BLK_N)

    // Tile the output tensor per SG and select tile for the current SG
    Tensor gD = local_tile(g_wg_D, take<0,2>(SubgroupTileShape{}), make_coord(m_sg,n_sg));            // (SG_M,SG_N)

    auto thread_xe_store_d = params.tiled_store_d.get_thread_slice(thread_idx);
    Tensor tCgD = thread_xe_store_d.partition_D(gD);

	  const int k_start_idx = crd2idx((*k_tile_iter), make_shape(params.k));
    int prefetch_k = k_start_idx;

    auto dequant = [](auto const& in, auto& out, auto& tCrS_input, const float* quant_map_) {
        constexpr auto N = decltype(cute::size<1>(in))::value;
        constexpr auto K = decltype(cute::size(out))::value / N;
    
        using compress_type = uint32_t;
        constexpr auto compress_size = cute::sizeof_bits_v<compress_type> / cute::sizeof_bits_v<ElementB>;
        static_assert((compress_size % N) == 0);
    
        constexpr auto vec_size = K / compress_size;
        using VecSrcType = cute::array<compress_type, vec_size>;
        using VecDstElemType = cute::array<ElementMMA, compress_size>;
        using VecDstType = cute::array<VecDstElemType, vec_size>;
    
        auto s_tensor = cute::make_tensor(
            (VecSrcType*)(cute::raw_pointer_cast(in.data())),
            cute::make_shape(cute::Int<K / (compress_size * vec_size)>{}, cute::Int<N>{})
        );
    
        auto d_tensor = cute::make_tensor(
            (VecDstType*)(cute::raw_pointer_cast(out.data())),
            cute::make_shape(cute::Int<K / (compress_size * vec_size)>{}, cute::Int<N>{})
        );
    
        #pragma unroll
        for (int n = 0; n < N; n++) {
            float ts = tCrS_input(n);
            auto& src = *(cute::array<VecSrcType, K / (compress_size * vec_size)>*)(s_tensor(_, n).data());
            auto& dst = *(cute::array<VecDstType, K / (compress_size * vec_size)>*)(d_tensor(_, n).data());
    
            #pragma unroll
            for (int k = 0; k < K / (compress_size * vec_size); k++) {
                VecDstType dst_val;
    
                #pragma unroll
                for (int i = 0; i < vec_size; i++) {
                    VecDstElemType dst_elem;
    
                    #pragma unroll
                    for (int j = 0; j < compress_size; j++) {
                        dst_elem[j] = static_cast<ElementMMA>(
                            quant_map_[(src[k][i] >> (4 * ((j+1)%2 + (j/2)*2))) & 0xf] * ts
                        );
                    }
                    dst_val[i] = dst_elem;
                }
                dst[k] = dst_val;
            }
        }
    };

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
      prefetch(tiled_prefetch_a, pAgA(_,_,_,prefetch_k));
      prefetch(tiled_prefetch_b, pBgB(_,_,_,prefetch_k));
    }

    for (int k_tile = k_start_idx, k_s = 0; k_tile < k_tile_count + k_start_idx; k_tile++, prefetch_k++, k_s++) {
      copy(params.tiled_copy_b, tBgB(_,_,_,k_tile), frag_copy_B);

      copy(params.tiled_copy_scale, tSgS(_, _, _, (k_start_idx + k_s) / k_reload_factor), frag_copy_Scale);

      dequant(dequant_frag, mma_B, fragment_scale, quant_map);

      copy(params.tiled_copy_a, tAgA(_,_,_,k_tile), frag_copy_A);

      if(prefetch_k < k_tile_count) {
        prefetch(tiled_prefetch_a, pAgA(_,_,_,prefetch_k));
        prefetch(tiled_prefetch_b, pBgB(_,_,_,prefetch_k));
      }

      cute::gemm(tiled_mma, mma_A, mma_B, accumulators);
      barrier_wait(3);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < FragsN; ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < FragsM; ++epi_m) {
         copy(params.tiled_store_d, accumulators(_, epi_m, epi_n), tCgD(_, epi_m, epi_n));        
      }
    }
  }
};

template <typename T, int BITS>
void gemm_4bit_cutlass(int m, int n, int k, int l, T *A, unsigned char *B,
                         float *absmax_, float *datatype, float *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream) {
  sycl::queue q = *stream;

  using GemmKernel = gemm_4bit_cutlass_kernel<T, BITS>;

  static constexpr int smem_size= 16*32/8; 

  auto problem_size = ProblemShape{m, n, k, l};

  using Params = GemmKernel::Params;
  Params params;
  params.m = m;
  params.n = n;
  params.k = k;
  params.l = l;
  params.A = A;
  params.B = B;
  params.out = out;
  params.datatype = datatype;
  params.group_size = blocksize;
  params.absmax = absmax_;
 
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  auto mA_mkl = make_tensor(make_gmem_ptr(A), make_layout(make_shape(m, k, l), stride_A));
  Copy_A tiled_copy_a{Copy_A{}.with(mA_mkl)};

  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  auto mB_nkl = make_tensor(cute::subbyte_iterator<ElementB>(B), make_layout(make_shape(n, k, l), stride_B));
  Copy_B tiled_copy_b{Copy_B{}.with(mB_nkl)};

  params.tiled_copy_a = tiled_copy_a;
  params.tiled_copy_b = tiled_copy_b;

  const int scale_k = cute::ceil_div(k, blocksize);
  StrideScale stride_S = cutlass::make_cute_packed_stride(StrideScale{}, cute::make_shape(n, scale_k, 1));
  auto mScale = make_tensor(make_gmem_ptr(absmax_), make_layout(make_shape(n, scale_k, 1), stride_S));
  Copy_Scale tiled_copy_scale{Copy_Scale{}.with(mScale)};

  params.tiled_copy_scale = tiled_copy_scale;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  auto problem_shape_MNKL = problem_size;

  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));
  auto mD = make_tensor(make_gmem_ptr(out), make_layout(make_shape(m, n, l), stride_D));
  Copy_D tiled_store_d = {tiled_store_d.with(mD)};
  params.tiled_store_d = tiled_store_d;

  params.hw_info = hw_info;

  TileSchedulerArguments scheduler{};
  params.scheduler = TileScheduler::to_underlying_arguments(
      problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, scheduler, nullptr);

  params.problem_shape = problem_size;


  dim3 const block = GemmKernel::get_block_shape();
  dim3 const grid = GemmKernel::get_grid_shape(params);

  const syclcompat::dim3 sycl_block(block.x, block.y, block.z);
  const syclcompat::dim3 sycl_grid(grid.x, grid.y, grid.z);
  //printf("Host Grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
  //printf("Host Block: (%d, %d, %d)\n", block.x, block.y, block.z);

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
}

template void gemm_4bit_cutlass<sycl::ext::oneapi::bfloat16, 16>(
    int m, int n, int k, int l, sycl::ext::oneapi::bfloat16 *A, unsigned char *B,
    float *absmax, float *datatype, float *out, int lda,
    int ldb, int ldc, int blocksize, sycl::queue *stream);

