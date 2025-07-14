#pragma once

#include <float.h>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include <cute/atom/copy_traits_xe.hpp>

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"

#include "cute/tensor.hpp"

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/util/type_traits.hpp>

#include <cute/atom/mma_traits_xe.hpp>

#include <sycl/sycl.hpp>
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"


#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor_predicate.hpp"

#include "cutlass/device_kernel.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/cuda_host_adapter.hpp"

#include "cutlass/kernel_launch.h"

// 3.x
#include "cutlass/util/sycl_event_manager.hpp"


template <typename T, int BITS>
void gemv_4bit_inference_cutlass_cute(int m, int n, int k, T *A, T *B,
                         float *absmax, float *datatype, float *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream);

template <typename T, int BITS>
void gemm_4bit_inference_cutlass_dequant(int m, int n, int k, T *A, unsigned char *B,
                         float *absmax, float *datatype, float *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream);

template <typename T, int BITS>
void gemv_4bit_inference_cutlass(int m, int n, int k, T *A, T *B,
                         float *absmax, float *datatype, float *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream);
