from collections.abc import Sequence
import ctypes as ct

import torch

from bitsandbytes.functional import _get_tensor_stream, get_ptr

from ..._ops import register_kernel
from ..utils import ipex_xpu
from ...cextension import lib

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
    print("this is bitsandbytes::_dequantize_4bit_impl ..")
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
            lib.cdequantize_blockwise_bf16_nf4(*args)
    elif dtype == torch.float16:
        #if quant_type == "fp4":
        #    lib.cdequantize_blockwise_fp16_fp4(*args)
        #else:
            lib.cdequantize_blockwise_fp16_nf4(*args)
    elif dtype == torch.float32:
        #if quant_type == "fp4":
        #    lib.cdequantize_blockwise_fp32_fp4(*args)
        #else:
            print("before lib.cdequantize_blockwise_fp32_nf4")
            lib.cdequantize_blockwise_fp32_nf4(*args)
            #lib.ctest(*args)
            print("after lib.cdequantize_blockwise_fp32_nf4")

@register_kernel("bitsandbytes::dequantize_4bit", "xpu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    print("this is bitsandbytes::dequantize_4bit ..")
    import pdb
    #pdb.set_trace()
    out = torch.zeros(shape, dtype=dtype, device=A.device)
    A_ref = A.clone()
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)

    ##just for kernel porting test, will replace by UT
    #out_ref = torch.ops.torch_ipex.dequantize_4bit(A_ref, "nf4", shape, absmax, None, blocksize).to(dtype)
    #max_diff = abs(out - out_ref)
    #print("max_diff = ", max_diff)
    #print(out[0])
    return out

if ipex_xpu: #will be replace by native kernel
    @register_kernel("bitsandbytes::dequantize_blockwise", "xpu")
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
