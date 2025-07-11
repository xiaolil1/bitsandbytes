#pytest tests/test_xpu.py
#pytest tests/test_functional.py::Test8BitBlockwiseQuantizeFunctional::test_dynamic_blockwise_quantization[signed=T-4096-nested=T-fp32-xpu]
#BNB_TEST_DEVICE="xpu" pytest tests/test_functional.py::Test8BitBlockwiseQuantizeFunctional
#BNB_TEST_DEVICE="xpu" pytest --ignore test_optim.py --ignore test_triton.py --ignore test_cuda_setup_evaluator.py
#
#BNB_TEST_DEVICE="xpu" pytest --ignore test_optim.py --ignore test_triton.py --ignore test_cuda_setup_evaluator.py tests/test_modules.py::test_embedding_lossless
#
#pytest -vs tests/test_xpu.py::TestXPU::test_gemv_4bit
#pytest -vs tests/test_xpu.py::TestXPU::test_4bit_quant
#pytest -vs tests/test_xpu.py::TestXPU::test_embedding_lossless

#pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=256-uint8-fp16-fc2-nf4-DQ_True-xpu]
#pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=256-fp16-fp16-fc2-nf4-DQ_True-xpu]
#pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=256-bf16-fp16-fc2-nf4-DQ_True-xpu]
#pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=256-fp32-fp16-fc2-nf4-DQ_True-xpu]
#pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=1024-uint8-bf16-attn_packed-nf4-DQ_True-xpu]
#pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=1024-uint8-bf16-attn_packed-nf4-DQ_False-xpu]
#pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=1024-fp16-bf16-attn_packed-nf4-DQ_True-xpu]
#pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=1024-fp16-bf16-attn_packed-nf4-DQ_False-xpu]
#pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=1024-bf16-bf16-attn_packed-nf4-DQ_True-xpu]
#pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=1024-bf16-bf16-attn_packed-nf4-DQ_False-xpu]
#pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=1024-fp32-bf16-attn_packed-nf4-DQ_True-xpu]
#pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=1024-fp32-bf16-attn_packed-nf4-DQ_False-xpu]


#pytest tests/test_functional.py::Test8BitBlockwiseQuantizeFunctional::test_dynamic_blockwise_quantization[signed=T-4096-nested=T-fp32-xpu]
#gdb -args python -m pytest tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=128-uint8-bf16-fc1-nf4-DQ_True-xpu]


#gdb -args python -m pytest -vs tests/test_xpu.py::TestXPU::test_gemm_4bit
python -m pytest -vs tests/test_xpu.py::TestXPU::test_gemm_4bit
