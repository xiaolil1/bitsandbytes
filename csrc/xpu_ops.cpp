////placeholder for test build
#include <vector>
#include <iostream>
#include <cmath>
#include <unistd.h>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

int main(void)
{
    // prepare device
    std::vector<sycl::device> root_devices;
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
       if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
          continue;
       }
       auto device_list = platform.get_devices();
       for (const auto& device : device_list) {
          if (device.is_gpu()) {
             root_devices.push_back(device);
          }
       }
    }
    auto first = root_devices[0];
    std::cout<<"device backend is: "<<first.get_backend()<<std::endl;
    sycl::device dev(first);
    sycl::queue q(dev, {sycl::property::queue::in_order()});

    //run_case<float>(q);

    return 0;
}
