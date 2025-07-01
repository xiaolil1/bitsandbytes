#include "xpu_cutlass.h"
#include <bit>
#include <cmath>
#include <iostream>

#include <sycl/sycl.hpp>

#if 0
template <typename T, size_t GROUP_SIZE, size_t NUM_PER_THREAD,
          size_t SUBG_SIZE, int BITS>
void kgemv_4bit_inference_cutlass<T, GROUP_SIZE, NUM_PER_THREAD, SUBG_SIZE,
                     BITS>::operator()(sycl::nd_item<1> item) const {
     std::cout<<"this is kgemv_4bit_inference_cutlass ...\n";
#if 0
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    // Complete the stride by combining static layout info (StrideA) with runtime size info (M,K,L)
    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_B.reset(static_cast<std::size_t>(K) * N * L);
    block_C.reset(static_cast<std::size_t>(M) * N * L);
    block_D.reset(static_cast<std::size_t>(M) * N * L);
    block_ref_D.reset(static_cast<std::size_t>(M) * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess){
      std::cout << "Invalid Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      std::exit(1);
    }

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    syclcompat::wait();
#endif
}

template class kgemv_4bit_inference_cutlass<sycl::half, 128, 4, 32, 16>;
template class kgemv_4bit_inference_cutlass<sycl::ext::oneapi::bfloat16, 128, 4, 32, 16>;
template class kgemv_4bit_inference_cutlass<float, 128, 4, 32, 32>;

#endif
template <typename T, int BITS>
void gemv_4bit_inference(int m, int n, int k, T *A, unsigned char *B,
                         float *absmax, float *datatype, T *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream) {
std::cout<<"this is gemv_4bit_inference cutlass...\n";
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
