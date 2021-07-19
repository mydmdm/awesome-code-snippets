#ifndef __DEVICEARRAY_CUH__
#define __DEVICEARRAY_CUH__

#include <cuda_runtime.h>

template <typename T>
class DeviceArray
{
public:
    DeviceArray(size_t s)
    {
        this->allocate(s);
    }

    ~DeviceArray()
    {
        this->free();
    }

    inline size_t size() const
    {
        return _end - _start;
    }

    inline const T *data(size_t offset = 0u) const
    {
        return _start + offset;
    }

    void from_host(const T *src)
    {
        auto result = cudaMemcpy(_start, src, this->size() * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to device memory");
        }
    }

    void to_host(T *dst)
    {
        auto result = cudaMemcpy(dst, _start, this->size() * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to host memory");
        }
    }

protected:
    void allocate(size_t size)
    {
        cudaError_t result = cudaMalloc((void **)&_start, size * sizeof(T));
        if (result != cudaSuccess)
        {
            _start = _end = nullptr;
            throw std::runtime_error("failed to allocate device memory");
        }
        size_ = size;
    }

    void free()
    {
        if (_start)
        {
            cudaFree(_start);
            _start = _end = nullptr;
        }
    }

    T *_start{nullptr};
    T *_end{nullptr};
};

#endif /* __DEVICEARRAY_CUH__ */
