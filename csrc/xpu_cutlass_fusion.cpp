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
using MmaType = sycl::ext::oneapi::bfloat16; //cutlass::bfloat16_t;
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

using TileShape = Shape<_64, _128, _64>;
using TiledMma =
    typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
                                  Layout<Shape<_2, _8, _1>, Stride<_8, _1, _0>>>::TiledMMA;
using GmemTiledCopyA = XE_2D_U16x16x32_LD_N;
using GmemTiledCopyB = XE_2D_U4x32x16_LD_T; 
constexpr int PipelineStages = 2;
static constexpr auto GROUP_SIZE=64; //Block Quant Size

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

//static constexpr auto FragsM = get<0>(SubgroupTileShape{}) / get<0>(MmaAtomShape());
//static constexpr auto FragsN = get<1>(SubgroupTileShape{}) / get<1>(MmaAtomShape());
//static constexpr auto FragmentSize = (get<0>(MmaAtomShape()) * get<1>(MmaAtomShape())) / SubgroupSize;
static constexpr int LUT_NUM = 4;

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
    float* quant_map_const;  

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

inline float dDequantizeNF4(unsigned char val) {

  // the values for this tree was generated by test_normal_map_tree
  // in the file tests/test_functional.py
  if ((val & 0b1000) == 8)
    if ((val & 0b0100) == 4)     // 1
      if ((val & 0b0010) == 2)   // 11
        if ((val & 0b0001) == 1) // 111
          return 1.0f;           //*1111
        else
          return 0.7229568362236023f; //*1110
      else if ((val & 0b0001) == 1)   // 110
        return 0.5626170039176941f;   //*1101
      else
        return 0.44070982933044434f; //*1100
    else if ((val & 0b0010) == 2)    // 10
      if ((val & 0b0001) == 1)       // 101
        return 0.33791524171829224f; //*1011
      else
        return 0.24611230194568634f; //*1010
    else if ((val & 0b0001) == 1)    // 100
      return 0.16093020141124725f;   //*1001
    else
      return 0.07958029955625534f; //*1000

  else if ((val & 0b0100) == 4) // 0
    if ((val & 0b0010) == 2)    // 01
      if ((val & 0b0001) == 1)  // 011
        return 0.0f;            //*0111
      else
        return -0.09105003625154495f; //*0110
    else if ((val & 0b0001) == 1)     // 010
      return -0.18477343022823334f;   //*0101
    else
      return -0.28444138169288635f; //*0100
  else if ((val & 0b0010) == 2)     // 00
    if ((val & 0b0001) == 1)        // 001
      return -0.39491748809814453f; //*0011
    else
      return -0.5250730514526367f; //*0010
  else if ((val & 0b0001) == 1)    // 000
    return -0.6961928009986877f;   //*0001
  else
    return -1.0f; //*0000
}

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    int thread_idx = int(ThreadIdxX());
	  const int m_coord = (params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN) 
                     ? BlockIdxY() : BlockIdxX();
    const int n_coord = (params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN) 
                     ? BlockIdxX() : BlockIdxY();
    const int l_coord = BlockIdxZ();

#if 1 
    // Load Dequatize LUT and save to SLM, 16 for 4bits
//    alignas(128) float* quant_map_ = reinterpret_cast<float*>(smem_buf);
//    if (thread_idx < 64) {
//      quant_map_[thread_idx] = params.datatype[thread_idx % 16]; 
//    }
//    barrier_arrive(3);
    //PVC SLM 64 banks -> 4 LUTs
    alignas(64) float (*quant_map_)[16] = reinterpret_cast<float(*)[16]>(smem_buf);
    if (thread_idx < 16 * LUT_NUM) {
      quant_map_[thread_idx / 16][thread_idx % 16] = params.datatype[thread_idx % 16]; 
    }
    barrier_arrive(3);
#else    
   static constexpr std::array<float, 16> quant_map_ = {
   // float __attribute__((opencl_private)) quant_map[16] = { 
        -1.0f, -0.6961928f, -0.52507305f, -0.39491749f,
        -0.28444138f, -0.18477343f, -0.09105004f, 0.0f,
        0.0795803f, 0.1609302f, 0.2461123f, 0.33791524f,
        0.44070983f, 0.562617f, 0.72295684f, 1.0f
    };
#endif
    Tensor mA_mkl = cute::get_pvc_tensor(make_shape(params.m, params.k, params.l));
    Tensor mB_nkl = cute::get_pvc_tensor(make_shape(params.n, params.k,1));
  
    Tensor gA = local_tile(mA_mkl, select<0,2>(TileShape{}), make_coord(m_coord,_,l_coord));
    Tensor gB = local_tile(mB_nkl, select<1,2>(TileShape{}), make_coord(n_coord,_,0));		
  
    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(TileShape{})); 
    clear(accumulators);

    auto k_tile_iter  = cute::make_coord_iterator(idx2crd(0, make_shape(params.k)), make_shape(params.k));

    auto thr_copy_A = params.tiled_copy_a.get_slice(thread_idx);
    auto thr_copy_B = params.tiled_copy_b.get_slice(thread_idx);
	  auto thr_copy_scale = params.tiled_copy_scale.get_slice(thread_idx);
  
    auto first_thread_in_sg_idx = syclcompat::get_nd_item<1>().get_sub_group().get_group_linear_id() * DispatchPolicy::SubgroupSize;
    auto sg_idx = syclcompat::get_nd_item<1>().get_sub_group().get_group_linear_id();
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx); 
  
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB); //values for each_thread (FrgV,(RestN,RestK),*)
	
    Tensor mma_A = make_tensor<ElementMMA>(make_fragment_layout(params.tiled_copy_a, tCgA(_,_,_,0).shape()));
    Tensor mma_B = make_tensor<ElementMMA>(make_fragment_layout(params.tiled_copy_b, tCgB(_,_,_,0).shape()));

#if 1 //SLM: 0, register: 1
  #if 1 //fragement register
	  Tensor dequant_frag = make_tensor<ElementB>(mma_B.layout());
  #else //common register
    using DequantLayout = Layout<Shape<_16, _1, _4>>;
	  Tensor dequant_frag = make_tensor<ElementB>(DequantLayout{});
  #endif  
    Tensor frag_copy_B = thr_copy_B.retile_D(dequant_frag);
#endif
    static constexpr auto scale_shape_t = decltype(size(typename GmemTiledCopyScale::BlockShape{}))::value / DispatchPolicy::SubgroupSize;
    static constexpr auto scale_shape_n = SG_QNT_WIDTH / decltype(size<1>(typename GmemTiledCopyScale::BlockShape{}))::value;
    static constexpr auto scale_shape_k = BLK_K / GROUP_SIZE < 1 ? 1 : BLK_K / GROUP_SIZE;
    using FragScaleLayout = Layout<Shape<Int<scale_shape_t>, Int<scale_shape_n>, Int<scale_shape_k>>>; //[1, dequant_N, block_num]
    Tensor fragment_scale = make_tensor<ElementScale>(FragScaleLayout{});
    
    //static_assert(std::is_same_v<typename decltype(dequant_frag)::value_type, ElementQuant>);
    static_assert(std::is_same_v<typename decltype(mma_A)::value_type, ElementMMA>);
    static_assert(std::is_same_v<typename decltype(mma_B)::value_type, ElementMMA>);
    //static_assert(params.group_size, GROUP_SIZE);

    Tensor frag_copy_A = thr_copy_A.retile_D(mma_A);
    //Tensor frag_copy_B = thr_copy_B.retile_D(dequant_frag);
    Tensor frag_copy_Scale = thr_copy_scale.retile_D(fragment_scale);

    Tensor tAgA = thr_copy_A.retile_S(tCgA);
    Tensor tBgB = thr_copy_B.retile_S(tCgB);

    auto tiled_prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>,Int<BLK_K>>, Num_SGs>(params.tiled_copy_a);
    auto tiled_prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>,Int<BLK_K>>, Num_SGs>(params.tiled_copy_b);
    auto thr_prefetch_A = tiled_prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = tiled_prefetch_b.get_slice(thread_idx);

    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);
	
	  const int k_tile_count = ceil_div(params.k, get<2>(WorkgroupTileShape{}));
    const int k_reload_factor = ceil_div(params.group_size, BLK_K);

    auto tSgS = [&](){
        return make_tensor(make_inttuple_iter(make_coord(n_coord * BLK_N + get<2>(thr_mma.thr_vmnk_)*SG_QNT_WIDTH, 0, 0)),
                          make_layout(make_shape(Int<scale_shape_t>{}, Int<scale_shape_n>{}, k_tile_count*scale_shape_k, k_tile_count * scale_shape_k),
                          make_stride(E<0>{}*(k_tile_count*scale_shape_n * scale_shape_k * DispatchPolicy::SubgroupSize), E<0>{}*(k_tile_count*scale_shape_k * DispatchPolicy::SubgroupSize), E<1>{}*_1{}, E<1>{}*_1{})));
//                          make_layout(make_shape(Int<scale_shape_t>{}, Int<scale_shape_n>{}, 1, k_tile_count * BLK_K/params.group_size),
//                          make_stride(E<0>{}*(scale_shape_n * k_tile_count * BLK_K/params.group_size), E<0>{}*(k_tile_count * BLK_K/params.group_size), E<0>{}*_0{}, E<1>{}*_1{})));
      
    }();
//    if(cute::thread0()) printf("scale_shape_t = %d, scale_shape_n = %d, scale_shape_k = %d, k_tile_count = %d, k_tile_count * BLK_K/params.group_size = %d, scale_shape_n * scale_shape_k * DispatchPolicy::SubgroupSize = %d, scale_shape_k * DispatchPolicy::SubgroupSize = %d\n",/*static_cast<int>(get<2>(thr_mma.thr_vmnk_)), static_cast<int>(SG_QNT_WIDTH),*/ scale_shape_t, scale_shape_n, scale_shape_k, k_tile_count, k_tile_count * BLK_K/params.group_size, scale_shape_n * scale_shape_k * DispatchPolicy::SubgroupSize, scale_shape_k * DispatchPolicy::SubgroupSize);

	  const int k_start_idx = crd2idx((*k_tile_iter), make_shape(params.k));
    int prefetch_k = k_start_idx;

#if 0 //SLM
  #if 1
  auto dequant = [&] (int k_tile) {
    constexpr int N = decltype(cute::size<1>(mma_B))::value;
    constexpr int K = decltype(cute::size(mma_B))::value / N;

    using src_compress_type = uint64_t;
    using dst_compress_type = uint64_t;
    constexpr int src_compress_size = cute::sizeof_bits_v<src_compress_type> / cute::sizeof_bits_v<ElementB>; //16
    constexpr int dst_compress_size = cute::sizeof_bits_v<dst_compress_type> / cute::sizeof_bits_v<ElementMMA>; //4
    constexpr int src_vec_size = (K / src_compress_size) >= 16 ? 16 : K / src_compress_size; //4, 16 -> max vec_size of sycl::vec
    constexpr int dst_vec_size = (K / dst_compress_size) >= 16 ? 16 : K / dst_compress_size; //16, 16 -> max vec_size of sycl::vec
    constexpr int src_loop_num = K / src_vec_size / src_compress_size;
    constexpr int dst_loop_num = K / dst_vec_size / dst_compress_size;

    alignas(8) ElementB* src = reinterpret_cast<ElementB*>(smem_buf) + thread_idx * K * 5; //for K=64, 4 is hardcode for 128B alignment.
    const uint8_t* gB_ptr = params.B + (n_coord * BLK_N + thread_idx * N) * params.k / 2 + k_tile * BLK_K / 2;
    ElementMMA* dst_slm = reinterpret_cast<ElementMMA*>(src + K); 
#if 0    
if(cute::thread0()) {
printf("src_compress_size = %d, dst_compress_size = %d, src_vec_size = %d, dst_vec_size = %d, src_loop_num = %d, dst_loop_num = %d\n", src_compress_size, dst_compress_size, src_vec_size, dst_vec_size, src_loop_num, dst_loop_num); 
//  print("\n\n======================= SLM: \n");
//      print("  src   : "); print(src);   print("\n");
//      print("  gB_ptr : "); print(gB_ptr); print("\n");
//      print("  dst_slm : "); print(dst_slm); print("\n");
//      print("   fragment_scale: "); print(fragment_scale); print("\n");
//  print("\n\n=======================\n\n");
}  
#endif
    #pragma unroll
    for (int n = 0; n < N; n++) {
      //float scale_value = fragment_scale(n);
      #pragma unroll
      for (int l = 0; l < src_loop_num; l++) {
        reinterpret_cast<sycl::vec<src_compress_type, src_vec_size>*>(src)[0] = reinterpret_cast<const sycl::vec<src_compress_type, src_vec_size>*>(gB_ptr)[n*src_loop_num + l];
        #pragma unroll
        for (int v = 0; v < src_vec_size; ++v) {
          src_compress_type src_value = reinterpret_cast<src_compress_type*>(src)[v];
          int dst_base_idx = l * src_vec_size * src_compress_size + v * src_compress_size;
          #pragma unroll
          for (int c = 0; c < src_compress_size; ++c) {
              uint8_t bit_value = (src_value >> (4 * (((c + 1) & 1) + (c >> 1) * 2))) & 0xF;
              float scale_value = fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + c) / GROUP_SIZE);
              dst_slm[dst_base_idx + c] = static_cast<ElementMMA>(quant_map[bit_value] * scale_value);
              //if(thread_idx==1 && m_coord==0 && n_coord==0 && l_coord==0) printf("dst_base_idx+c = %d, n * (BLK_K / GROUP_SIZE) + (dst_base_idx+c)/GROUP_SIZE) = %d, scale_value = %f\n",dst_base_idx+c, n * (BLK_K / GROUP_SIZE) + (dst_base_idx+c)/GROUP_SIZE, scale_value);
          }
        }
      }
      
      #pragma unroll
      for (int l = 0; l < dst_loop_num; l++) {
        reinterpret_cast<sycl::vec<dst_compress_type, dst_vec_size>*>(cute::raw_pointer_cast(mma_B.data()))[n*dst_loop_num + l] = reinterpret_cast<const sycl::vec<dst_compress_type, dst_vec_size>*>(dst_slm)[l];
      }
    }
  };
  #endif
#else //register
#if 0
      auto dequant = [&] (float* quant_map){
        constexpr int N = decltype(cute::size<1>(mma_B))::value;
        constexpr int K = decltype(cute::size(mma_B))::value / N;

//int index = 2;
//sycl::vec<float, 3> q_vec = {1,2,3};
//auto mask = (sycl::vec<int, 3>(0,1,2) == index);
//float value = sycl::reduce(q_vec * mask.convert<float>());

        using src_compress_type = uint32_t;
        using dst_compress_type = uint64_t;
        constexpr int src_compress_size = 8; //cute::sizeof_bits_v<src_compress_type> / cute::sizeof_bits_v<ElementB>; //16
        constexpr int dst_compress_size = 4; //cute::sizeof_bits_v<dst_compress_type> / cute::sizeof_bits_v<ElementMMA>; //4
        constexpr int src_vec_size = 8; //(K / src_compress_size) >= 16 ? 16 : K / src_compress_size; //4, 16 -> max vec_size of sycl::vec
        constexpr int dst_vec_size = 8; //(K / dst_compress_size) >= 16 ? 16 : K / dst_compress_size; //16, 16 -> max vec_size of sycl::vec
        constexpr int src_loop_num = 1; //K / src_vec_size / src_compress_size;
        constexpr int dst_loop_num = 2; //K / dst_vec_size / dst_compress_size;

        //src_compress_type src[src_loop_num * src_vec_size];
        //ElementMMA dst[dst_loop_num * dst_compress_size];
        ElementMMA dst[dst_loop_num * dst_compress_size * dst_vec_size];

        //ElementMMA dst[dst_loop_num * dst_compress_size];

        //reinterpret_cast<sycl::vec<src_compress_type, src_vec_size>*>(src)[0] = reinterpret_cast<sycl::vec<src_compress_type, src_vec_size>*>(cute::raw_pointer_cast(dequant_frag.data()))[0];
        float scale_value = fragment_scale(0);//(dst_base_idx + c) >> (31 - std::countl_zero<unsigned int>(GROUP_SIZE)));

        src_compress_type src_1, src_2;
        //int map_offset = 16*(sg_idx%4); 
        int v = 0;
        src_1 = reinterpret_cast<src_compress_type*>(cute::raw_pointer_cast(dequant_frag.data()))[v];
        
        #pragma unroll
        for (; v < src_vec_size-1;) {
          src_2 = src_1;
          int dst_base_idx = v * src_compress_size;
          //int map_offset = dst_base_idx % 2 * 16;
          v++;
          src_1 = reinterpret_cast<src_compress_type*>(cute::raw_pointer_cast(dequant_frag.data()))[v];
          int c = 0;
          uint8_t bit_value = (src_2 >> (4 * (((c + 1) & 1) + (c >> 1) * 2))) & 0xF;
          float converted_value_1 = quant_map[bit_value + (dst_base_idx + c) % 4 * 16];
          float converted_value_2 = 0.f;
          #pragma unroll
          for (; c < src_compress_size-1;) {
              converted_value_2 = converted_value_1;
              c++;
              bit_value = (src_2 >> (4 * (((c + 1) & 1) + (c >> 1) * 2))) & 0xF;
              converted_value_1 = quant_map[bit_value + (dst_base_idx + c - 1) % 4 * 16];
              dst[dst_base_idx + c-1] = static_cast<ElementMMA>(converted_value_2 * scale_value);
          }
          dst[dst_base_idx + c] = static_cast<ElementMMA>(converted_value_1 * scale_value);
          //reinterpret_cast<sycl::vec<dst_compress_type, 1>*>(cute::raw_pointer_cast(mma_B.data()))[2*(v-1)] = reinterpret_cast<sycl::vec<dst_compress_type, 1>*>(dst)[0];
          //reinterpret_cast<sycl::vec<dst_compress_type, 1>*>(cute::raw_pointer_cast(mma_B.data()))[2*(v-1)+1] = reinterpret_cast<sycl::vec<dst_compress_type, 1>*>(dst)[1];
          //reinterpret_cast<dst_compress_type*>(cute::raw_pointer_cast(mma_B.data()))[2*(v-1)] = reinterpret_cast<dst_compress_type*>(dst)[0];
          //reinterpret_cast<dst_compress_type*>(cute::raw_pointer_cast(mma_B.data()))[2*(v-1)+1] = reinterpret_cast<dst_compress_type*>(dst)[1];
          reinterpret_cast<sycl::vec<dst_compress_type, dst_loop_num>*>(cute::raw_pointer_cast(mma_B.data()))[v-1] = reinterpret_cast<sycl::vec<dst_compress_type, dst_loop_num>*>(dst)[v-1];
        }
        src_2 = src_1;
        int dst_base_idx = v * src_compress_size;
        //int map_offset = dst_base_idx % 2 * 16;
        int c = 0;
        uint8_t bit_value = (src_2 >> (4 * (((c + 1) & 1) + (c >> 1) * 2))) & 0xF;
        float converted_value_1 = quant_map[bit_value + (dst_base_idx + c) % 4 * 16];
        float converted_value_2 = 0.f;
        #pragma unroll
        for (; c < src_compress_size-1;) {
            converted_value_2 = converted_value_1;
            c++;
            bit_value = (src_2 >> (4 * (((c + 1) & 1) + (c >> 1) * 2))) & 0xF;
            converted_value_1 = quant_map[bit_value + (dst_base_idx + c - 1) % 4 * 16];
            dst[dst_base_idx + c-1] = static_cast<ElementMMA>(converted_value_2 * scale_value);
        }
        dst[dst_base_idx + c] = static_cast<ElementMMA>(converted_value_1 * scale_value);
        //reinterpret_cast<sycl::vec<dst_compress_type, 1>*>(cute::raw_pointer_cast(mma_B.data()))[2*v] = reinterpret_cast<sycl::vec<dst_compress_type, 1>*>(dst)[0];
        //reinterpret_cast<sycl::vec<dst_compress_type, 1>*>(cute::raw_pointer_cast(mma_B.data()))[2*v+1] = reinterpret_cast<sycl::vec<dst_compress_type, 1>*>(dst)[1];
        reinterpret_cast<sycl::vec<dst_compress_type, dst_loop_num>*>(cute::raw_pointer_cast(mma_B.data()))[v] = reinterpret_cast<sycl::vec<dst_compress_type, dst_loop_num>*>(dst)[v];
        
//        reinterpret_cast<sycl::vec<src_compress_type, src_vec_size>*>(src)[1] = reinterpret_cast<sycl::vec<src_compress_type, src_vec_size>*>(cute::raw_pointer_cast(dequant_frag.data()))[1];
//        scale_value = fragment_scale(1);
        //reinterpret_cast<sycl::vec<dst_compress_type, dst_vec_size>*>(cute::raw_pointer_cast(mma_B.data()))[0] = reinterpret_cast<sycl::vec<dst_compress_type, dst_vec_size>*>(dst)[0];

//        #pragma unroll
//        for (int v = src_vec_size; v < src_loop_num * src_vec_size; v++) {
//          int dst_base_idx = v * src_compress_size;
//          int c = 0;
//          uint8_t bit_value = (src[v] >> (4 * (((c + 1) & 1) + (c >> 1) * 2))) & 0xF;
//          float converted_value_1 = quant_map[bit_value];
//          float converted_value_2 = 0.f;
//          #pragma unroll
//          for (; c < src_compress_size-1;) {
//              converted_value_2 = converted_value_1;
//              c++;
//              bit_value = (src[v] >> (4 * (((c + 1) & 1) + (c >> 1) * 2))) & 0xF;
//              converted_value_1 = quant_map[bit_value];
//              dst[dst_base_idx + c-1] = static_cast<ElementMMA>(converted_value_2 * scale_value);
//          }
//          dst[dst_base_idx + c] = static_cast<ElementMMA>(converted_value_1 * scale_value);
//        }
//
//        reinterpret_cast<sycl::vec<dst_compress_type, dst_vec_size>*>(cute::raw_pointer_cast(mma_B.data()))[1] = reinterpret_cast<sycl::vec<dst_compress_type, dst_vec_size>*>(dst)[1];

      };
#else
      auto dequant = [&] (int start_lut_id){
        constexpr int N = decltype(cute::size<1>(mma_B))::value;
        constexpr int K = decltype(cute::size(mma_B))::value / N;
  
        using src_compress_type = uint8_t;
        using dst_compress_type = uint32_t;
        constexpr int src_compress_size = cute::sizeof_bits_v<src_compress_type> / cute::sizeof_bits_v<ElementB>; //16
        constexpr int dst_compress_size = cute::sizeof_bits_v<dst_compress_type> / cute::sizeof_bits_v<ElementMMA>; //4
        constexpr int src_vec_size = 4; //(K / src_compress_size) >= 16 ? 16 : K / src_compress_size; //4, 16 -> max vec_size of sycl::vec
        constexpr int dst_vec_size = 4; //(K / dst_compress_size) >= 16 ? 16 : K / dst_compress_size; //16, 16 -> max vec_size of sycl::vec
        constexpr int src_loop_num = K / src_vec_size / src_compress_size;
        constexpr int dst_loop_num = K / dst_vec_size / dst_compress_size;
        src_compress_type src[src_vec_size];
        ElementMMA dst[dst_loop_num * dst_compress_size * dst_vec_size];

//if(cute::thread0()) {
//printf("src_compress_size = %d, dst_compress_size = %d, src_vec_size = %d, dst_vec_size = %d, src_loop_num = %d, dst_loop_num = %d\n", src_compress_size, dst_compress_size, src_vec_size, dst_vec_size, src_loop_num, dst_loop_num);
//}
        int lut_id = start_lut_id;
//if(sg_idx == 0){
//  for (int i = 0; i < 64; i++){
//    printf("tid = %d, dequant_frag ptr[%d] = %x, mma_B ptr[%d] = %x\n",thread_idx, i, cute::raw_pointer_cast(dequant_frag.data()+i),i, cute::raw_pointer_cast(mma_B.data()+i));
//  }
//}

        int pre_num = 1;
        #pragma unroll
        for (int n = 0; n < N; n++) {
          #pragma unroll
          for (int l = 0; l < src_loop_num; l++) {
            reinterpret_cast<sycl::vec<src_compress_type, src_vec_size>*>(src)[0] = reinterpret_cast<sycl::vec<src_compress_type, src_vec_size>*>(cute::raw_pointer_cast(dequant_frag.data()))[n*src_loop_num + l];

            #pragma unroll
            for (int v = 0; v < src_vec_size; v++) {
              //src_compress_type src_value = reinterpret_cast<sycl::vec<src_compress_type, src_vec_size>*>(cute::raw_pointer_cast(dequant_frag.data()))[n*src_loop_num + l][v]; //src[v];
              src_compress_type src_value = src[v];
              int dst_base_idx = l * src_vec_size * src_compress_size + v * src_compress_size;
#if 0              
              int c = 0;
              uint16_t high_bits_1 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2 + 1))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c) / GROUP_SIZE)));
              lut_id = (lut_id + 1) % LUT_NUM;
              uint16_t low_bits_1 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c + 1) / GROUP_SIZE)));

              c++;
              reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(mma_B.data()))[n*src_loop_num*src_compress_size/2 + l * src_vec_size*src_compress_size/2 + v*src_compress_size/2 + c-pre_num] = (static_cast<uint32_t>(low_bits_1) << 16) | high_bits_1;
              uint16_t high_bits_2 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2 + 1))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c) / GROUP_SIZE)));
              lut_id = (lut_id + 1) % LUT_NUM;
              uint16_t low_bits_2 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c + 1) / GROUP_SIZE)));

              c++;
              reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(mma_B.data()))[n*src_loop_num*src_compress_size/2 + l * src_vec_size*src_compress_size/2 + v*src_compress_size/2 + c-pre_num] = (static_cast<uint32_t>(low_bits_2) << 16) | high_bits_2;
              uint16_t high_bits_3 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2 + 1))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c) / GROUP_SIZE)));
              lut_id = (lut_id + 1) % LUT_NUM;
              uint16_t low_bits_3 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c + 1) / GROUP_SIZE)));

              c++;
              reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(mma_B.data()))[n*src_loop_num*src_compress_size/2 + l * src_vec_size*src_compress_size/2 + v*src_compress_size/2 + c-pre_num] = (static_cast<uint32_t>(low_bits_3) << 16) | high_bits_3;
              uint16_t high_bits_4 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2 + 1))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c) / GROUP_SIZE)));
              lut_id = (lut_id + 1) % LUT_NUM;
              uint16_t low_bits_4 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c + 1) / GROUP_SIZE)));

              c++;
              reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(mma_B.data()))[n*src_loop_num*src_compress_size/2 + l * src_vec_size*src_compress_size/2 + v*src_compress_size/2 + c-pre_num] = (static_cast<uint32_t>(low_bits_4) << 16) | high_bits_4;
              uint16_t high_bits_5 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2 + 1))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c) / GROUP_SIZE)));
              lut_id = (lut_id + 1) % LUT_NUM;
              uint16_t low_bits_5 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c + 1) / GROUP_SIZE)));

              c++;
              reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(mma_B.data()))[n*src_loop_num*src_compress_size/2 + l * src_vec_size*src_compress_size/2 + v*src_compress_size/2 + c-pre_num] = (static_cast<uint32_t>(low_bits_5) << 16) | high_bits_5;
              uint16_t high_bits_6 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2 + 1))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c) / GROUP_SIZE)));
              lut_id = (lut_id + 1) % LUT_NUM;
              uint16_t low_bits_6 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c + 1) / GROUP_SIZE)));

              c++;
              reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(mma_B.data()))[n*src_loop_num*src_compress_size/2 + l * src_vec_size*src_compress_size/2 + v*src_compress_size/2 + c-pre_num] = (static_cast<uint32_t>(low_bits_6) << 16) | high_bits_6;
              uint16_t high_bits_7 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2 + 1))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c) / GROUP_SIZE)));
              lut_id = (lut_id + 1) % LUT_NUM;
              uint16_t low_bits_7 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c + 1) / GROUP_SIZE)));

              c++;
              reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(mma_B.data()))[n*src_loop_num*src_compress_size/2 + l * src_vec_size*src_compress_size/2 + v*src_compress_size/2 + c-1] = (static_cast<uint32_t>(low_bits_7) << 16) | high_bits_7;
              uint16_t high_bits_8 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2 + 1))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c) / GROUP_SIZE)));
              lut_id = (lut_id + 1) % LUT_NUM;
              uint16_t low_bits_8 = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c + 1) / GROUP_SIZE)));
              reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(mma_B.data()))[n*src_loop_num*src_compress_size/2 + l * src_vec_size*src_compress_size/2 + v*src_compress_size/2 + c] = (static_cast<uint32_t>(low_bits_8) << 16) | high_bits_8;
#elif 0
              for (int c = 0; c < src_compress_size/2; c++) {
                uint16_t high_bits = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2 + 1))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c) / GROUP_SIZE)));
                lut_id = (lut_id + 1) % LUT_NUM;
                uint16_t low_bits = sycl::bit_cast<uint16_t>(static_cast<ElementMMA>(quant_map_[lut_id][(src_value >> (4 * (c * 2))) & 0xf] * fragment_scale(n * (BLK_K / GROUP_SIZE) + (dst_base_idx + 2 * c + 1) / GROUP_SIZE)));
                reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(mma_B.data()))[n*src_loop_num*src_compress_size/2 + l * src_vec_size*src_compress_size/2 + v*src_compress_size/2 + c] = (static_cast<uint32_t>(low_bits) << 16) | high_bits;
              }
#elif 0
              #pragma unroll
              for (int c = 0; c < src_compress_size; c++) {
                uint8_t bit_value = (src_value >> (4 * (((c + 1) & 1) + (c >> 1) * 2))) & 0xF;
                float scale_value = fragment_scale((n * BLK_K  + dst_base_idx + c) >> (31 - std::countl_zero<unsigned int>(GROUP_SIZE)));
                dst[dst_base_idx + c] = static_cast<ElementMMA>(quant_map_[lut_id][bit_value] * scale_value);
                lut_id = (lut_id + 1) % LUT_NUM;
              }
              //reinterpret_cast<sycl::vec<dst_compress_type, 1>*>(cute::raw_pointer_cast(mma_B.data()))[n*src_loop_num + l * src_vec_size + v] = reinterpret_cast<sycl::vec<dst_compress_type, 1>*>(dst)[l * src_vec_size + v];
            }
              //reinterpret_cast<sycl::vec<dst_compress_type, dst_vec_size>*>(cute::raw_pointer_cast(mma_B.data()))[n*src_loop_num + l] = reinterpret_cast<sycl::vec<dst_compress_type, dst_vec_size>*>(dst)[l];
          }
          #pragma unroll
          for (int l = 0; l < dst_loop_num; l++) {
            reinterpret_cast<sycl::vec<dst_compress_type, dst_vec_size>*>(cute::raw_pointer_cast(mma_B.data()))[n * dst_loop_num + l] = reinterpret_cast<sycl::vec<dst_compress_type, dst_vec_size>*>(dst)[l];
            //reinterpret_cast<dst_compress_type*>(cute::raw_pointer_cast(mma_B.data()))[n * dst_loop_num + l] = reinterpret_cast<dst_compress_type*>(dst)[l];
            //reinterpret_cast<sycl::vec<dst_compress_type, 2>*>(cute::raw_pointer_cast(mma_B.data()))[n*dst_loop_num + l] = reinterpret_cast<sycl::vec<dst_compress_type, 2>*>(dst)[l];

          }          
#else
               #pragma unroll
              for (int c = 0; c < src_compress_size; c++) {
                  uint8_t bit_value = (src_value >> (4 * (((c + 1) & 1) + (c >> 1) * 2))) & 0xF;
                  float scale_value = fragment_scale((n * BLK_K  + dst_base_idx + c) >> (31 - std::countl_zero<unsigned int>(GROUP_SIZE)));
                  dst[dst_base_idx + c] = static_cast<ElementMMA>(quant_map_[lut_id][bit_value] * scale_value);
                  lut_id = (lut_id + 1) % LUT_NUM;
              }
            }
          }

          #pragma unroll
          for (int l = 0; l < dst_loop_num; l++) {
            reinterpret_cast<sycl::vec<dst_compress_type, dst_vec_size>*>(cute::raw_pointer_cast(mma_B.data()))[n * dst_loop_num + l] = reinterpret_cast<sycl::vec<dst_compress_type, dst_vec_size>*>(dst)[l];
          }
#endif              
        }
      };
#endif
#endif

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
      prefetch(tiled_prefetch_a, pAgA(_,_,_,prefetch_k));
      prefetch(tiled_prefetch_b, pBgB(_,_,_,prefetch_k));
    }

    //int map_offset = 16 * (sg_idx % 4);
    //int map_offset = 16 * ((sg_idx ^ (sg_idx >> 2)) % 4);
    //int lut_id = sg_idx % 4;
    int start_lut_id = sg_idx % LUT_NUM;
    //int start_lut_id = (sg_idx + (sg_idx >> 2)) & 3;
    //printf("sg_idx = %d, start_lut_id = %d\n", sg_idx, start_lut_id);

    for (int k_tile = k_start_idx, k_s = 0; k_tile < k_tile_count; k_tile++, k_s++, prefetch_k++) {
#if 1 //SLM: 0, register: 1     
      copy(params.tiled_copy_b, tBgB(_,_,_,k_tile), frag_copy_B);
      copy(params.tiled_copy_scale, tSgS(_, _, _, (k_start_idx + k_s) * BLK_K/params.group_size), frag_copy_Scale);
      copy(params.tiled_copy_a, tAgA(_,_,_,k_tile), frag_copy_A);
      //dequant((sg_idx % 4 ) < 2 ? quant_map_1 : quant_map_2);
      //dequant(quant_map_ + map_offset);
      //dequant(quant_map_[lut_id]);
      dequant(start_lut_id);
#else
      copy(params.tiled_copy_scale, tSgS(_, _, _, (k_start_idx + k_s) * BLK_K/params.group_size), frag_copy_Scale);
      copy(params.tiled_copy_a, tAgA(_,_,_,k_tile), frag_copy_A);
      dequant(k_tile);
#endif      
      if (prefetch_k < k_tile_count) {
        prefetch(tiled_prefetch_a, pAgA(_,_,_,prefetch_k));
        prefetch(tiled_prefetch_b, pBgB(_,_,_,prefetch_k));
      }
      
      cute::gemm(tiled_mma, mma_A, mma_B, accumulators);
      barrier_wait(3);
    }

//replace epilige for store
//    Tensor mD_mnl = cute::get_pvc_tensor(make_shape(params.m, params.n, params.l));
//    Tensor g_wg_D = local_tile(mD_mnl, take<0,2>(WorkgroupTileShape{}), make_coord(m_coord,n_coord,l_coord));
//    Tensor gD = local_tile(g_wg_D, take<0,2>(SubgroupTileShape{}), make_coord(
//      get_sub_group_id() / ATOM_N, 
//      get_sub_group_id() % ATOM_N
//    ));
//    
//    auto thread_xe_store_d = params.tiled_store_d.get_thread_slice(thread_idx);
//    Tensor tCgD = thread_xe_store_d.partition_D(gD);

    static constexpr int FragsM = get<0>(SubgroupTileShape{}) / get<0>(MmaAtomShape()); // atom numbers per thread; A frags per sub_group
    static constexpr int FragsN = get<1>(SubgroupTileShape{}) / get<1>(MmaAtomShape()); // atom numbers per thread; B frags per sub_group

    auto m_sg = get_sub_group_id() / ATOM_N;
    auto n_sg = get_sub_group_id() % ATOM_N;

    Tensor mD_mnl = cute::get_pvc_tensor(make_shape(params.m, params.n, params.l)); // Logical full output tensor

    // Tile the output tensor per WG and select the tile for current WG
    Tensor g_wg_D = local_tile(mD_mnl, take<0,2>(TileShape{}), make_coord(m_coord,n_coord,l_coord));

    // Tile the output tensor per SG and select tile for the current SG
    Tensor gD = local_tile(g_wg_D, take<0,2>(SubgroupTileShape{}), make_coord(m_sg,n_sg));

    auto thread_xe_store_d = params.tiled_store_d.get_thread_slice(thread_idx); //partial copy_atom for current thread
    Tensor tCgD = thread_xe_store_d.partition_D(gD); //values for current thread

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
  //std::cout<<"group_size = "<<blocksize<<std::endl;

#if 1
  static constexpr int smem_size= (16) * sizeof(float) * LUT_NUM;
  //static constexpr int smem_size= (16) * sizeof(float) * LUT_NUM + BLK_N * BLK_K * sizeof(ElementMMA)*2;
#else  
  static constexpr int smem_size = BLK_N * BLK_K * sizeof(ElementMMA) * 2 * 2; //aligned with 128B and will be reused for dequant src and dst.
#endif  
  size_t max_slm_size = q.get_device().get_info<sycl::info::device::local_mem_size>();
  assert(smem_size <= max_slm_size);

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
  Copy_D tiled_store_d = {Copy_D{}.with(mD)};
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


  float quant_map[16] = {
      -1.0f, -0.6961928f, -0.52507305f, -0.39491749f,
      -0.28444138f, -0.18477343f, -0.09105004f, 0.0f,
      0.0795803f, 0.1609302f, 0.2461123f, 0.33791524f,
      0.44070983f, 0.562617f, 0.72295684f, 1.0f
  };

  syclcompat::constant_memory<float, 1> const_mem(16);
  syclcompat::memcpy(const_mem.get_ptr(), quant_map, sizeof(float) * 16);



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
  
  float* const_mem_ptr = const_mem.get_ptr();
  params.quant_map_const = const_mem_ptr;
  syclcompat::experimental::launch<device_kernel<GemmKernel>>(policy, q, params);//, const_mem.get_ptr());
}

template void gemm_4bit_cutlass<sycl::ext::oneapi::bfloat16, 16>(
    int m, int n, int k, int l, sycl::ext::oneapi::bfloat16 *A, unsigned char *B,
    float *absmax, float *datatype, float *out, int lda,
    int ldb, int ldc, int blocksize, sycl::queue *stream);

