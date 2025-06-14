from collections.abc import Sequence
import ctypes as ct

import torch
import pdb
from bitsandbytes.functional import _get_tensor_stream, get_ptr

from ..._ops import register_kernel
from ..utils import ipex_xpu
from ...cextension import lib

import pdb

if torch.__version__ >= (2, 7):
    @register_kernel("bitsandbytes::int8_linear_matmul", "xpu")
    def _(A: torch.Tensor, B: torch.Tensor):
        return torch._int_mm(
            A.reshape(-1, A.shape[-1]),
            B.t(),
        ).reshape(*A.shape[:-1], B.shape[0])

def _dequantize_4bit_impl(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    dtype: torch.dtype,
    out: torch.Tensor,
    ) -> None:
    import pdb
    #pdb.set_trace()
    #print("this is bitsandbytes::_dequantize_4bit_impl ..")
    args = (
        None,
        get_ptr(A),
        get_ptr(absmax),
        get_ptr(out),
        ct.c_int(blocksize),
        ct.c_int(out.numel()),
        _get_tensor_stream(A),
    )

    #lib.cdequantize_blockwise_fp32_nf4(*args)
    if dtype == torch.bfloat16:
        #if quant_type == "fp4":
        #    lib.cdequantize_blockwise_bf16_fp4(*args)
        #else:
            #print("before lib.cdequantize_blockwise_bf16_nf4")
            #pdb.set_trace()
            lib.cdequantize_blockwise_bf16_nf4(*args)
            #pdb.set_trace()
            #print("after lib.cdequantize_blockwise_bf16_nf4")
    elif dtype == torch.float16:
        #if quant_type == "fp4":
        #    lib.cdequantize_blockwise_fp16_fp4(*args)
        #else:
            #print("before lib.cdequantize_blockwise_fp16_nf4")
            #pdb.set_trace()
            lib.cdequantize_blockwise_fp16_nf4(*args)
            #pdb.set_trace()
            #print("after lib.cdequantize_blockwise_fp16_nf4")
            #pdb.set_trace()
    elif dtype == torch.float32:
        #if quant_type == "fp4":
        #    lib.cdequantize_blockwise_fp32_fp4(*args)
        #else:
            #print("before lib.cdequantize_blockwise_fp32_nf4")
            #pdb.set_trace()
            lib.cdequantize_blockwise_fp32_nf4(*args)
            #pdb.set_trace()
            #lib.ctest(*args)
            #print("after lib.cdequantize_blockwise_fp32_nf4")

@register_kernel("bitsandbytes::dequantize_4bit", "xpu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    #print("this is bitsandbytes::dequantize_4bit ..")
    import pdb
    out = torch.zeros(shape, dtype=dtype, device=A.device)
    A_ref = A.clone()
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)

    ##just for kernel porting test, will replace by UT
    #out_ref = torch.ops.torch_ipex.dequantize_4bit(A_ref, "nf4", shape, absmax, None, blocksize).to(dtype)
    #max_diff = abs(out - out_ref)
    #print("max_diff = ", max_diff)
    #print(out[0])
    return out

#if ipex_xpu: #will be replace by native kernel
#    @register_kernel("bitsandbytes::dequantize_blockwise", "xpu")
#    def _(
#        A: torch.Tensor,
#        absmax: torch.Tensor,
#        code: torch.Tensor,
#        blocksize: int,
#        dtype: torch.dtype,
#    ) -> torch.Tensor:
#        shape = A.shape
#        out = torch.empty(A.reshape(-1).shape, dtype=dtype, device=A.device)
#        # void cdequantize_blockwise_fp32(
#        # float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, cudaStream_t stream)
#        if dtype == torch.float16:
#            ipex_xpu.xpu.bitsandbytes.cdequantize_blockwise_fp16(code, A, absmax, out, blocksize, A.numel())
#        elif dtype == torch.bfloat16:
#            ipex_xpu.xpu.bitsandbytes.cdequantize_blockwise_bf16(code, A, absmax, out, blocksize, A.numel())
#        elif dtype == torch.float32:
#            ipex_xpu.xpu.bitsandbytes.cdequantize_blockwise_fp32(code, A, absmax, out, blocksize, A.numel())
#        else:
#            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {out.dtype}")
#
#        return out.reshape(shape)
@register_kernel("bitsandbytes::dequantize_blockwise", "xpu")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    out = torch.empty_like(A, dtype=dtype)
    _dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out=out)
    return out


@register_kernel("bitsandbytes::dequantize_blockwise.out", "xpu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")
    torch._check(out.shape == A.shape, lambda: f"Expected out.shape == {A.shape}, got {out.shape}")
    _dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out=out)


def _dequantize_blockwise_impl(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype, out: torch.Tensor
) -> None:
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(
        dtype in [torch.float16, torch.bfloat16, torch.float32],
        lambda: f"Blockwise dequantization only supports 16bit/32bit floating types, got {dtype}",
    )

    args = (
        get_ptr(code),
        get_ptr(A),
        get_ptr(absmax),
        get_ptr(out),
        ct.c_int(blocksize),
        ct.c_int(A.numel()),
        _get_tensor_stream(A),
    )
    #pdb.set_trace()
    if dtype == torch.float16:
        lib.cdequantize_blockwise_fp16(*args)
    elif dtype == torch.bfloat16:
        lib.cdequantize_blockwise_bf16(*args)
    elif dtype == torch.float32:
         lib.cdequantize_blockwise_fp32(*args)

@register_kernel("bitsandbytes::gemv_4bit", "xpu")
def _(
    A: torch.Tensor, B: torch.Tensor, shapeB: Sequence[int], absmax: torch.Tensor, code: torch.Tensor, blocksize: int
) -> torch.Tensor:
    shape = (*A.shape[:-1], shapeB[0])
    out = torch.empty(shape, device=A.device, dtype=A.dtype)
    #pdb.set_trace()
    _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out=out)
    #pdb.set_trace()
    return out


@register_kernel("bitsandbytes::gemv_4bit.out", "xpu")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    out: torch.Tensor,
) -> None:
    torch._check(
        out.shape == (*A.shape[:-1], shapeB[0]),
        lambda: f"Expected out.shape == {(*A.shape[:-1], shapeB[0])}, got {out.shape}",
    )
    torch._check(out.dtype == A.dtype, lambda: f"Expected out.dtype == {A.dtype}, got {out.dtype}")
    _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out=out)


def _gemv_4bit_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    out: torch.Tensor,
) -> None:
    #torch._check_is_size(blocksize)
    #torch._check(
    #    A.numel() == A.size(-1),
    #    lambda: f"A must be a vector with leading dimensions of 1, got {A.shape}",
    #)
    #torch._check(
    #    A.dtype in [torch.float16, torch.bfloat16, torch.float32],
    #    lambda: f"A must be float16, bfloat16, or float32, got {A.dtype}",
    #)
    #torch._check(
    #    B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32],
    #    lambda: f"B must be backed by storage of type uint8, bfloat16, float16, or float32, got {B.dtype}",
    #)
    #torch._check(absmax.dtype == torch.float32, lambda: f"absmax must be float32, got {absmax.dtype}")
    #torch._check(code.dtype == torch.float32, lambda: f"code must be float32, got {code.dtype}")

    m = ct.c_int32(shapeB[0])
    n = ct.c_int32(1)
    k = ct.c_int32(shapeB[1])

    lda = m
    ldb = ct.c_int32((A.shape[-1] + 1) // 2)
    ldc = m

    stream = _get_tensor_stream(A)

    #print("this is _gemv_4bit_impl ....")
    #pdb.set_trace()

    if A.dtype == torch.float16:
        lib.cgemm_4bit_inference_fp16(
            m,
            n,
            k,
            get_ptr(A),
            get_ptr(B),
            get_ptr(absmax),
            get_ptr(code),
            get_ptr(out),
            lda,
            ldb,
            ldc,
            ct.c_int32(blocksize),
            stream,
        )
    elif A.dtype == torch.bfloat16:
        lib.cgemm_4bit_inference_bf16(
            m,
            n,
            k,
            get_ptr(A),
            get_ptr(B),
            get_ptr(absmax),
            get_ptr(code),
            get_ptr(out),
            lda,
            ldb,
            ldc,
            ct.c_int32(blocksize),
            stream,
        )
    elif A.dtype == torch.float32:
        lib.cgemm_4bit_inference_fp32(
            m,
            n,
            k,
            get_ptr(A),
            get_ptr(B),
            get_ptr(absmax),
            get_ptr(code),
            get_ptr(out),
            lda,
            ldb,
            ldc,
            ct.c_int32(blocksize),
            stream,
        )

@register_kernel("bitsandbytes::dequantize_nf4_ipex", "xpu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    import pdb
    pdb.set_trace()
    out = torch.ops.torch_ipex.dequantize_4bit(A, "nf4", shape, absmax, None, blocksize).t().to(dtype)
    return out

@register_kernel("bitsandbytes::dequantize_blockwise_ipex", "xpu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    shape = A.shape
    out = torch.empty(A.reshape(-1).shape, dtype=dtype, device=A.device)
    # void cdequantize_blockwise_fp32(
    # float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, cudaStream_t stream)
    if dtype == torch.float16:
        ipex_xpu.xpu.bitsandbytes.cdequantize_blockwise_fp16(code, A, absmax, out, blocksize, A.numel())
    elif dtype == torch.bfloat16:
        ipex_xpu.xpu.bitsandbytes.cdequantize_blockwise_bf16(code, A, absmax, out, blocksize, A.numel())
    elif dtype == torch.float32:
        ipex_xpu.xpu.bitsandbytes.cdequantize_blockwise_fp32(code, A, absmax, out, blocksize, A.numel())
    else:
        raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {out.dtype}")

    return out.reshape(shape)        
