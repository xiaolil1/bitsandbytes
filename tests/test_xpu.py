import math
import random
import time

import einops
import numpy as np
import pytest
import torch

import bitsandbytes
import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import (
    BOOLEAN_TUPLES,
    TRUE_FALSE,
    describe_dtype,
    get_available_devices,
    get_test_dims,
    id_formatter,
)

import pdb

torch.set_printoptions(precision=5, sci_mode=False, linewidth=120, edgeitems=20, threshold=10000)
#k = 20
def assert_all_approx_close(a, b, rtol=1e-3, atol=1e-3, count=0, throw=True):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    sumval = (idx == 0).sum().item()
    if sumval > count:
        if throw:
            print(f"Too many values not close: assert {sumval} < {count}")
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

    return sumval

class TestQuantize4Bit:
    @pytest.mark.parametrize("device", ["xpu"]) #get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float32], ids=describe_dtype) #, torch.float16, torch.bfloat16], ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4"]) #["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512, 1024, 2048, 4096])
    def test_4bit_quant(self, device, dtype, quant_type, blocksize):
        A1 = torch.randn(1024, 512, device=device, dtype=dtype)
        #A1 = torch.ones(64, device=device, dtype=dtype)
        qa, SA = F.quantize_4bit(A1, blocksize=blocksize, quant_type=quant_type)
        #pdb.set_trace()
        A2 = F.dequantize_4bit(qa, SA, blocksize=blocksize, quant_type=quant_type)
        #print("A1 = ", A1)
        #print("A2 = ", A2)
        #pdb.set_trace()

        err = (A1 - A2).abs().float()
        relerr = (err / (A1.abs().float() + 1e-8)).mean()
        err = err.mean()

        assert A2.dtype == dtype

        # With larger block sizes, we can expect this to blow up.
        # At blocksize>=1024, don't even bother looking at relerr.
        if blocksize <= 64:
            assert err.item() < 0.1
            assert relerr.item() < 0.28
        elif blocksize <= 256:
            assert err.item() < 0.11
            assert relerr.item() < 0.30
        elif blocksize <= 512:
            assert err.item() < 0.12
            assert relerr.item() < 0.31
        elif quant_type == "fp4":
            # 1024 => 0.48, 2048 => 0.52, 4096 => 0.56
            assert err.item() < 0.08 + math.log2(blocksize) * 4e-2
        else:
            # 1024 => 0.8, 2048 => 0.88, 4096 => 0.96
            assert err.item() < math.log2(blocksize) * 8e-2

    @pytest.mark.parametrize("device", ["xpu"])#get_available_devices())
    @pytest.mark.parametrize("double_quant", ["False"], ids=lambda double_quant: f"DQ_{double_quant}") #]TRUE_FALSE, ids=lambda double_quant: f"DQ_{double_quant}")
    @pytest.mark.parametrize("storage_type", ["nf4"]) #, "fp4"])
    @pytest.mark.parametrize("kind", ["fc1", "fc2", "attn"])#, "attn_packed"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
    @pytest.mark.parametrize(
        "quant_storage",
        [torch.uint8, torch.float16, torch.bfloat16, torch.float32],
        ids=describe_dtype,
    )
    @pytest.mark.parametrize("dim", [256, 512, 1024], ids=id_formatter("dim"))
    def test_gemv_4bit(self, device, dim, dtype, storage_type, quant_storage, double_quant, kind):
        errs1 = []
        errs2 = []
        errs3 = []
        relerrs1 = []
        relerrs2 = []
        relerrs3 = []
        max_errs1 = []
        max_errs2 = []
        max_errs3 = []

        # Large number of iterations is excessive and slow on CPU.
        # Keep for CUDA for now.
        iters = 1 if device == "cuda" else 10

        for i in range(iters):
            #pdb.set_trace()
            if kind == "fc1":
                A = torch.randn(1, dim, dtype=dtype, device=device)
                B = torch.randn(dim * 4, dim, dtype=dtype, device=device) / math.sqrt(dim)
            elif kind == "fc2":
                A = torch.randn(1, 4 * dim, dtype=dtype, device=device)
                B = torch.randn(dim, 4 * dim, dtype=dtype, device=device) / math.sqrt(dim)
            elif kind == "attn":
                A = torch.randn(1, dim, dtype=dtype, device=device)
                B = torch.randn(dim, dim, dtype=dtype, device=device) / math.sqrt(dim)
            elif kind == "attn_packed":
                A = torch.randn(1, dim, dtype=dtype, device=device)
                B = torch.randn(dim * 3, dim, dtype=dtype, device=device) / math.sqrt(dim)

            qB, state = F.quantize_4bit(
                B,
                quant_type=storage_type,
                compress_statistics=double_quant,
                quant_storage=quant_storage,
            )
            #pdb.set_trace()
            C3 = torch.matmul(A, B.t())
            #pdb.set_trace()
            C2 = F.gemv_4bit(A, qB.t(), state=state)
            #A.requires_grad = True
            C1 = bnb.matmul_4bit(A, qB.t(), state)

            #pdb.set_trace()
            err1 = (C1 - C2).abs().float()
            err2 = (C3 - C2).abs().float()
            #print("err2 = ",err2)
            err3 = (C3 - C1).abs().float()

            mag1 = torch.abs(C1).float() + 1e-5
            mag2 = torch.abs(C3).float() + 1e-5
            mag3 = torch.abs(C3).float() + 1e-5

            relerr1 = err1 / mag1
            relerr2 = err2 / mag2
            relerr3 = err3 / mag3

            max_err1 = err1.max()
            max_err2 = err2.max()
            max_err3 = err3.max()

            errs1.append(err1.mean().item())
            errs2.append(err2.mean().item())
            errs3.append(err3.mean().item())

            relerrs1.append(relerr1.mean().item())
            relerrs2.append(relerr2.mean().item())
            relerrs3.append(relerr3.mean().item())

            max_errs1.append(max_err1.item())
            max_errs2.append(max_err2.item())
            max_errs3.append(max_err3.item())

            c = int(C1.numel() * 0.0014 * (dim / 256)) + 1

            c = assert_all_approx_close(C1, C2, 1e-5, 0.01, count=0, throw=False)
        err1 = sum(errs1) / len(errs1) / math.sqrt(dim)
        err2 = sum(errs2) / len(errs2) / math.sqrt(dim)
        err3 = sum(errs3) / len(errs3) / math.sqrt(dim)
        relerr1 = sum(relerrs1) / len(relerrs1) / math.sqrt(dim)
        relerr2 = sum(relerrs2) / len(relerrs2) / math.sqrt(dim)
        relerr3 = sum(relerrs3) / len(relerrs3) / math.sqrt(dim)
        maxerr1 = sum(max_errs1) / len(max_errs1) / math.sqrt(dim)
        maxerr2 = sum(max_errs2) / len(max_errs2) / math.sqrt(dim)
        maxerr3 = sum(max_errs3) / len(max_errs3) / math.sqrt(dim)
        absratio = err2 / err3
        relratio = relerr2 / relerr3
        maxratio = relerr2 / relerr3

        # for debugging if the tests fails
        #
        # print('='*80)
        # print(f'For matmul: {A.shape}, {B.shape}, {kind}, {dtype}, {storage_type}, double_quant={double_quant}:')
        # print(C1.flatten()[-20:])
        # print(C2.flatten()[-20:])
        # print(f'inference vs training abs: {err1}')
        # print(f'inference vs training rel: {relerr1}')
        # print(f'inference vs training max: {maxerr1}')
        # print(f'inference vs training vs torch err ratio abs: {absratio}')
        # print(f'inference vs training vs torch err ratio rel: {relratio}')
        # print(f'inference vs training vs torch err ratio max: {maxratio}')
        if dtype == torch.float16:
            if dim <= 512:
                assert err1 < 7e-5

                # TODO(matthewdouglas): On T4, dim=128-fp16-fc2-fp4-DQ will have relerror ~ 0.00092727
                if (
                    device == "cuda"
                    and double_quant
                    and storage_type == "fp4"
                    and kind == "fc2"
                    and torch.cuda.get_device_capability() == (7, 5)
                ):
                    assert relerr1 < 0.00093
                else:
                    assert relerr1 < 0.0008
            else:
                assert err1 < 6e-5
                assert relerr1 < 2e-4
            assert absratio < 1.005 and absratio > 0.995
            assert relratio < 1.005 and relratio > 0.995
            assert maxratio < 1.005 and maxratio > 0.995
        elif dtype == torch.float32:
            if dim <= 512:
                assert err1 < 5e-8
                assert relerr1 < 1e-6
                assert maxerr1 < 1e-7
            else:
                assert err1 < 5e-8
                assert relerr1 < 8e-6
                assert maxerr1 < 1e-7
            assert absratio < 1.005 and absratio > 0.995
            assert relratio < 1.005 and relratio > 0.995
            assert maxratio < 1.005 and maxratio > 0.995
        elif dtype == torch.bfloat16:
            if dim <= 512:
                assert err1 < 6e-4
                assert relerr1 < 0.007
                assert maxerr1 < 0.015
            else:
                assert err1 < 2e-4
                assert relerr1 < 0.002
                assert maxerr1 < 0.0012
            assert absratio < 1.005 and absratio > 0.995
            assert relratio < 1.04 and relratio > 0.96
            assert maxratio < 1.02 and maxratio > 0.98

    @pytest.mark.parametrize("device", ["xpu"]) #get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16], ids=id_formatter("storage_dtype"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512])
    def test_gemv_4bit_op(self, device, dtype, storage_dtype, quant_type, blocksize):
        out_features = 1024
        in_features = 256

        A = torch.randn((1, 1, in_features), dtype=dtype, device=device)
        B = torch.randn((out_features, in_features), dtype=dtype, device=A.device)
        B_q, absmax = torch.ops.bitsandbytes.quantize_4bit(B, blocksize, quant_type, storage_dtype)
        code = bitsandbytes.functional.get_4bit_type(quant_type, device=A.device, blocksize=blocksize)

        out = torch.ops.bitsandbytes.gemv_4bit(A, B_q, B.shape, absmax, code, blocksize)

        assert out.device == A.device
        assert out.dtype == dtype
        assert out.shape == (1, 1, out_features)
        assert out.isreal().all()

        torch.library.opcheck(torch.ops.bitsandbytes.gemv_4bit.default, (A, B_q, B.shape, absmax, code, blocksize))
