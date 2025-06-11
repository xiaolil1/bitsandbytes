import math
import random
import time

import einops
import numpy as np
import pytest
import torch

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

torch.set_printoptions(precision=5, sci_mode=False, linewidth=120, edgeitems=20, threshold=10000)
#k = 20

class TestQuantize4BitFunctional:
    @pytest.mark.parametrize("device", ["xpu"]) #get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float32]) #, torch.float16, torch.bfloat16], ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4"]) #["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [64]) #, 128, 256, 512, 1024, 2048, 4096])
    def test_4bit_quant(self, device, dtype, quant_type, blocksize):
        A1 = torch.randn(1024, 512, device=device, dtype=dtype)
        qa, SA = F.quantize_4bit(A1, blocksize=blocksize, quant_type=quant_type)
        A2 = F.dequantize_4bit(qa, SA, blocksize=blocksize, quant_type=quant_type)
        #print("A1 = ", A1[0])
        #print("A2 = ", A2[0])

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
