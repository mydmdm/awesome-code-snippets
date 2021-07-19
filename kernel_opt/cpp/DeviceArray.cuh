#ifndef __DEVICEARRAY_CUH__
#define __DEVICEARRAY_CUH__

#include "HostArray.h"

#include <cuda_runtime.h>
#include <stdexcept>

template <typename T>
struct DeviceAllocator
{
    void allocate(size_t size, T **p_start, T **p_end)
    {
        cudaError_t result = cudaMalloc((void **)p_start, size * sizeof(T));
        if (result != cudaSuccess)
        {
            *p_start = nullptr;
            *p_end = nullptr;
            throw std::runtime_error("failed to allocate device memory");
        }
        *p_end = (*p_start) + size;
    }
    void deallocate(T **p_start, T **p_end)
    {
        if (*p_start)
        {
            cudaFree(*p_start);
            *p_start = nullptr;
            *p_end = nullptr;
        }
    }
};

template <typename T>
class DeviceArray : public ArrayBase<T, DeviceAllocator<T>>
{
    using ArrayBase<T, DeviceAllocator<T>>::ArrayBase;

    void from_host(const T *src)
    {
        auto result = cudaMemcpy(this->_start, src, this->size() * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to device memory");
        }
    }

    void to_host(T *dst)
    {
        auto result = cudaMemcpy(dst, this->_start, this->size() * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to host memory");
        }
    }
};

#endif /* __DEVICEARRAY_CUH__ */
