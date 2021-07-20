#ifndef __DEVICEARRAY_CUH__
#define __DEVICEARRAY_CUH__

#include "HostArray.h"

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
