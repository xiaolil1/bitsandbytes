#make clean
cmake -DCOMPUTE_BACKEND=xpu -S . -B build -DCMAKE_CXX_COMPILER=/home/gta/intel/oneapi/compiler/2025.1/bin/icpx -DCMAKE_C_COMPILER=/home/gta/intel/oneapi/compiler/2025.1/bin/icx -DCMAKE_BUILD_TYPE=Debug
cd build
make
cd ..
pip install -e .
