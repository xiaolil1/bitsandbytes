cmake -DCOMPUTE_BACKEND=xpu -S .
#cmake -DCOMPUTE_BACKEND=xpu -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-O0 -g" -S .
make
pip install -e .
