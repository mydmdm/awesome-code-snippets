#ifndef __UTILS_CUH__
#define __UTILS_CUH__

#include <cuda_runtime.h>
#include <stdexcept>

inline void check_cuda(cudaError_t result, const char *msg = nullptr)
{
    if (result != cudaSuccess)
    {
        if (msg)
        {
            throw std::runtime_error(msg);
        }
        else
        {
            throw std::runtime_error("operation failed");
        }
    }
}

void print_device_properties(int device = 0)
{
    cudaDeviceProp prop;
    check_cuda(cudaGetDeviceProperties(&prop, device));
    fprintf(stdout, "Properties of NVIDIA GPU %d (cm_%d%d)\n", device, prop.major, prop.minor);
    fprintf(stdout, " Global memory available on device in bytes: %lu\n", prop.totalGlobalMem);
    fprintf(stdout, " Shared memory available per block in bytes: %lu\n", prop.sharedMemPerBlock);
    fprintf(stdout, " 32-bit registers available per block: %d\n", prop.regsPerBlock);
    fprintf(stdout, " Warp size in threads: %d\n", prop.warpSize);
}

#endif /* __UTILS_CUH__ */
