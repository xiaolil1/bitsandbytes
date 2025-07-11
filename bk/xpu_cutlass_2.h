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

//#include "cutlass/cutlass.h"
//#include "cutlass/gemm/dispatch_policy.hpp"
//#include "cutlass/gemm/gemm.h"
//#include "cutlass/kernel_hardware_info.hpp"
//
//#include "cute/algorithm/functional.hpp"
//#include "cute/atom/mma_atom.hpp"
//#include "cute/algorithm/gemm.hpp"
//#include "cute/tensor_predicate.hpp"

using namespace cute;

template <typename T, int BITS>
void gemv_4bit_fusion(int m, int n, int k, T *A, unsigned char *B,
                         float *absmax, float *datatype, T *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream);
