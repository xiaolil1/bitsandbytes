/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>
#include <cute/util/sycl_vec.hpp>

#ifdef __SYCL_DEVICE_ONLY__ 
#define SYCL_DEVICE_OCL(x) SYCL_EXTERNAL x
#else 
#define SYCL_DEVICE_OCL(x) inline x { CUTE_INVALID_CONTROL_PATH("Trying to use XE built-in on non-XE hardware"); }
#endif

// mma_bf16
SYCL_DEVICE_OCL(cute::intel::float8 intel_sub_group_bf16_bf16_matrix_mad_k16(cute::intel::short8 a, cute::intel::int8 b, cute::intel::float8 acc));
SYCL_DEVICE_OCL(cute::intel::float4 intel_sub_group_bf16_bf16_matrix_mad_k16(cute::intel::short4 a, cute::intel::int8 b, cute::intel::float4 acc));
SYCL_DEVICE_OCL(cute::intel::float2 intel_sub_group_bf16_bf16_matrix_mad_k16(cute::intel::short2 a, cute::intel::int8 b, cute::intel::float2 acc));
SYCL_DEVICE_OCL(float  intel_sub_group_bf16_bf16_matrix_mad_k16(short a, cute::intel::int8 b, float acc));
// mma_half
SYCL_DEVICE_OCL(cute::intel::float8 intel_sub_group_f16_f16_matrix_mad_k16(cute::intel::short8 a, cute::intel::int8 b, cute::intel::float8 acc));
SYCL_DEVICE_OCL(cute::intel::float4 intel_sub_group_f16_f16_matrix_mad_k16(cute::intel::short4 a, cute::intel::int8 b, cute::intel::float4 acc));
SYCL_DEVICE_OCL(cute::intel::float2 intel_sub_group_f16_f16_matrix_mad_k16(cute::intel::short2 a, cute::intel::int8 b, cute::intel::float2 acc));
SYCL_DEVICE_OCL(float  intel_sub_group_f16_f16_matrix_mad_k16(short a, cute::intel::int8 b, float acc));
// mma_s8
SYCL_DEVICE_OCL(cute::intel::int8 intel_sub_group_i8_i8_matrix_mad_k32(cute::intel::short8 a, cute::intel::int8 b, cute::intel::int8 acc));
SYCL_DEVICE_OCL(cute::intel::int4 intel_sub_group_i8_i8_matrix_mad_k32(cute::intel::short4 a, cute::intel::int8 b, cute::intel::int4 acc));
SYCL_DEVICE_OCL(cute::intel::int2 intel_sub_group_i8_i8_matrix_mad_k32(cute::intel::short2 a, cute::intel::int8 b, cute::intel::int2 acc));
SYCL_DEVICE_OCL(int  intel_sub_group_i8_i8_matrix_mad_k32(short a, cute::intel::int8 b, int acc));
// mma_u8
SYCL_DEVICE_OCL(cute::intel::int8 intel_sub_group_u8_u8_matrix_mad_k32(cute::intel::ushort8 a, cute::intel::uint8 b, cute::intel::int8 acc));
SYCL_DEVICE_OCL(cute::intel::int4 intel_sub_group_u8_u8_matrix_mad_k32(cute::intel::ushort4 a, cute::intel::uint8 b, cute::intel::int4 acc));
SYCL_DEVICE_OCL(cute::intel::int2 intel_sub_group_u8_u8_matrix_mad_k32(cute::intel::ushort2 a, cute::intel::uint8 b, cute::intel::int2 acc));
SYCL_DEVICE_OCL(int  intel_sub_group_u8_u8_matrix_mad_k32(cute::intel::ushort a, cute::intel::uint8 b, int acc));
// mma_tf32
SYCL_DEVICE_OCL(cute::intel::float8 intel_sub_group_tf32_tf32_matrix_mad_k8(cute::intel::float4 a, cute::intel::float8 b, cute::intel::float8 acc));
SYCL_DEVICE_OCL(cute::intel::float4 intel_sub_group_tf32_tf32_matrix_mad_k8(cute::intel::float2 a, cute::intel::float8 b, cute::intel::float4 acc));
SYCL_DEVICE_OCL(cute::intel::float2 intel_sub_group_tf32_tf32_matrix_mad_k8(float a, cute::intel::float8 b, cute::intel::float2 acc));
SYCL_DEVICE_OCL(float  intel_sub_group_tf32_tf32_matrix_mad_k8(float a, cute::intel::float8 b, float acc));

#undef SYCL_DEVICE_OCL

namespace cute {
//MxNxK_D,A,B,C
//# of vector component of a x subgroup-size x function name
//float8 intel_sub_group_bf16_bf16_matrix_mad_k16(short8 a, int8 b, float8 acc);
//TODO: Is A really not transposed? Maybe better a macro than separate define for 1,2,4,8
struct XE_8x16x16_F32BF16BF16F32_TT
{
  using DRegisters = intel::float8[1];
  using ARegisters = intel::short8[1];
  using BRegisters = intel::int8[1];
  using CRegisters = intel::float8[1];

  CUTE_HOST_DEVICE static void
  fma(intel::float8      & d,
      intel::short8 const& a,
      intel::int8   const& b,
      intel::float8 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_8x16x16_F32BF16BF16F32_TT on non-PVC hardware");
#endif
  }
};
struct XE_4x16x16_F32BF16BF16F32_TT
{
  using DRegisters = intel::float4[1];
  using ARegisters = intel::short4[1];
  using BRegisters = intel::int8[1];
  using CRegisters = intel::float4[1];

  CUTE_HOST_DEVICE static void
  fma(intel::float4      & d,
      intel::short4 const& a,
      intel::int8   const& b,
      intel::float4 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_8x16x16_F32BF16BF16F32_TT on non-PVC hardware");
#endif
  }
};
struct XE_2x16x16_F32BF16BF16F32_TT
{
  using DRegisters = intel::float2[1];
  using ARegisters = intel::short2[1];
  using BRegisters = intel::int8[1];
  using CRegisters = intel::float2[1];

  CUTE_HOST_DEVICE static void
  fma(intel::float2      & d,
      intel::short2 const& a,
      intel::int8   const& b,
      intel::float2 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_8x16x16_F32BF16BF16F32_TT on non-PVC hardware");
#endif
  }
};
//float  intel_sub_group_bf16_bf16_matrix_mad_k16(short  a, int8 b, float  acc)
struct XE_1x16x16_F32BF16BF16F32_TT
{
  using DRegisters = float[1];
  using ARegisters = short[1];
  using BRegisters = intel::int8[1];
  using CRegisters = float[1];

  CUTE_HOST_DEVICE static void
  fma(float      & d,
      short const& a,
      intel::int8  const& b,
      float const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_1x16x16_F32BF16BF16F32_TT on non-PVC hardware");
#endif
  }
};

//MxNxK_D,A,B,C
//# of vector component of a x subgroup-size x function name
//float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, int8 acc);
//TODO: Is A really not transposed? Maybe better a macro than separate define for 1,2,4,8
struct XE_8x16x16_F32F16F16F32_TT
{
  using DRegisters = intel::float8[1];
  using ARegisters = intel::short8[1];
  using BRegisters = intel::int8[1];
  using CRegisters = intel::float8[1];

  CUTE_HOST_DEVICE static void
  fma(intel::float8      & d,
      intel::short8 const& a,
      intel::int8   const& b,
      intel::float8 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_8x16x16_F32F16F16F32_TT on non-PVC hardware");
#endif
  }
};

struct XE_4x16x16_F32F16F16F32_TT
{
  using DRegisters = intel::float4[1];
  using ARegisters = intel::short4[1];
  using BRegisters = intel::int8[1];
  using CRegisters = intel::float4[1];

  CUTE_HOST_DEVICE static void
  fma(intel::float4      & d,
      intel::short4 const& a,
      intel::int8   const& b,
      intel::float4 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_4x16x16_F32F16F16F32_TT on non-PVC hardware");
#endif
  }
};

struct XE_2x16x16_F32F16F16F32_TT
{
  using DRegisters = intel::float2[1];
  using ARegisters = intel::short2[1];
  using BRegisters = intel::int8[1];
  using CRegisters = intel::float2[1];

  CUTE_HOST_DEVICE static void
  fma(intel::float2      & d,
      intel::short2 const& a,
      intel::int8   const& b,
      intel::float2 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_2x16x16_F32F16F16F32_TT on non-PVC hardware");
#endif
  }
};

struct XE_1x16x16_F32F16F16F32_TT
{
  using DRegisters = float[1];
  using ARegisters = short[1];
  using BRegisters = intel::int8[1];
  using CRegisters = float[1];

  CUTE_HOST_DEVICE static void
  fma(float      & d,
      short const& a,
      intel::int8  const& b,
      float const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_1x16x16_F32F16F16F32_TT on non-PVC hardware");
#endif
  }
};

//MxNxK_A,B,C,D
//# of vector component of a x subgroup-size x function name
//float8 intel_sub_group_i8_i8_matrix_mad_k16(short8 a, int8 b, float8 acc);
//TODO: Is A really not transposed? Maybe better a macro than separate define for 1,2,4,8
struct XE_8x16x32_S32S8S8S32_TT
{
  using DRegisters = intel::int8[1];
  using ARegisters = intel::short8[1];
  using BRegisters = intel::int8[1];
  using CRegisters = intel::int8[1];

  CUTE_HOST_DEVICE static void
  fma(intel::int8      & d,
      intel::short8 const& a,
      intel::int8   const& b,
      intel::int8 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_i8_i8_matrix_mad_k32(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_8x16x32_S32S8S8S32_TT on non-PVC hardware");
#endif
  }
};

struct XE_4x16x32_S32S8S8S32_TT
{
  using DRegisters = intel::int4[1];
  using ARegisters = intel::short4[1];
  using BRegisters = intel::int8[1];
  using CRegisters = intel::int4[1];

  CUTE_HOST_DEVICE static void
  fma(intel::int4      & d,
      intel::short4 const& a,
      intel::int8   const& b,
      intel::int4 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_i8_i8_matrix_mad_k32(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_4x16x32_S32S8S8S32_TT on non-PVC hardware");
#endif
  }
};

struct XE_2x16x32_S32S8S8S32_TT
{
  using DRegisters = intel::int2[1];
  using ARegisters = intel::short2[1];
  using BRegisters = intel::int8[1];
  using CRegisters = intel::int2[1];

  CUTE_HOST_DEVICE static void
  fma(intel::int2      & d,
      intel::short2 const& a,
      intel::int8   const& b,
      intel::int2 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_i8_i8_matrix_mad_k32(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_2x16x32_S32S8S8S32_TT on non-PVC hardware");
#endif
  }
};

struct XE_1x16x32_S32S8S8S32_TT
{
  using DRegisters = int[1];
  using ARegisters = short[1];
  using BRegisters = intel::int8[1];
  using CRegisters = int[1];

  CUTE_HOST_DEVICE static void
  fma(int      & d,
      short const& a,
      intel::int8  const& b,
      int const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_i8_i8_matrix_mad_k32(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_1x16x32_S32S8S8S32_TT on non-PVC hardware");
#endif
  }
};

struct XE_8x16x32_S32U8U8S32_TT
{
  using DRegisters = intel::int8[1];
  using ARegisters = intel::ushort8[1];
  using BRegisters = intel::uint8[1];
  using CRegisters = intel::int8[1];

  CUTE_HOST_DEVICE static void
  fma(intel::int8      & d,
      intel::ushort8 const& a,
      intel::uint8   const& b,
      intel::int8 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_u8_u8_matrix_mad_k32(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_8x16x32_S32U8U8S32_TT on non-PVC hardware");
#endif
  }
};

struct XE_4x16x32_S32U8U8S32_TT
{
  using DRegisters = intel::int4[1];
  using ARegisters = intel::ushort4[1];
  using BRegisters = intel::uint8[1];
  using CRegisters = intel::int4[1];

  CUTE_HOST_DEVICE static void
  fma(intel::int4      & d,
      intel::ushort4 const& a,
      intel::uint8   const& b,
      intel::int4 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_u8_u8_matrix_mad_k32(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_4x16x32_S32U8U8S32_TT on non-PVC hardware");
#endif
  }
};

struct XE_2x16x32_S32U8U8S32_TT
{
  using DRegisters = intel::int2[1];
  using ARegisters = intel::ushort2[1];
  using BRegisters = intel::uint8[1];
  using CRegisters = intel::int2[1];

  CUTE_HOST_DEVICE static void
  fma(intel::int2      & d,
      intel::ushort2 const& a,
      intel::uint8   const& b,
      intel::int2 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_u8_u8_matrix_mad_k32(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_2x16x32_S32U8U8S32_TT on non-PVC hardware");
#endif
  }
};

struct XE_1x16x32_S32U8U8S32_TT
{
  using DRegisters = int[1];
  using ARegisters = cute::intel::ushort[1];
  using BRegisters = intel::uint8[1];
  using CRegisters = int[1];

  CUTE_HOST_DEVICE static void
  fma(int      & d,
      cute::intel::ushort const& a,
      intel::uint8  const& b,
      int const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_u8_u8_matrix_mad_k32(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_1x16x32_S32U8U8S32_TT on non-PVC hardware");
#endif
  }
};

struct XE_8x16x8_F32TF32TF32F32_TT
{
  using DRegisters = intel::float8[1];
  using ARegisters = intel::float4[1];
  using BRegisters = intel::float8[1];
  using CRegisters = intel::float8[1];

  CUTE_HOST_DEVICE static void
  fma(intel::float8      & d,
      intel::float4 const& a,
      intel::float8   const& b,
      intel::float8 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_tf32_tf32_matrix_mad_k8(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_8x16x8_F32TF32TF32F32_TT on non-PVC hardware");
#endif
  }
};

struct XE_4x16x8_F32TF32TF32F32_TT
{
  using DRegisters = intel::float4[1];
  using ARegisters = intel::float2[1];
  using BRegisters = intel::float8[1];
  using CRegisters = intel::float4[1];

  CUTE_HOST_DEVICE static void
  fma(intel::float4      & d,
      intel::float2 const& a,
      intel::float8   const& b,
      intel::float4 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_tf32_tf32_matrix_mad_k8(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_4x16x8_F32TF32TF32F32_TT on non-PVC hardware");
#endif
  }
};

struct XE_2x16x8_F32TF32TF32F32_TT
{
  using DRegisters = intel::float2[1];
  using ARegisters = float[1];
  using BRegisters = intel::float8[1];
  using CRegisters = intel::float2[1];

  CUTE_HOST_DEVICE static void
  fma(intel::float2      & d,
      float const& a,
      intel::float8   const& b,
      intel::float2 const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_tf32_tf32_matrix_mad_k8(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_2x16x8_F32TF32TF32F32_TT on non-PVC hardware");
#endif
  }
};
//float  intel_sub_group_bf16_bf16_matrix_mad_k16(short  a, int8 b, float  acc)
struct XE_1x16x8_F32TF32TF32F32_TT
{
  using DRegisters = float[1];
  using ARegisters = float[1];
  using BRegisters = intel::float8[1];
  using CRegisters = float[1];

  CUTE_HOST_DEVICE static void
  fma(float      & d,
      float const& a,
      intel::float8  const& b,
      float const& c)
  {
#if defined(SYCL_INTEL_TARGET)
    d = intel_sub_group_tf32_tf32_matrix_mad_k8(a, b, c);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use XE_1x16x8_F32TF32TF32F32_TT on non-PVC hardware");
#endif
  }
};
} //namespace cute
