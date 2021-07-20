#ifndef __ALLOCATOR_CUH__
#define __ALLOCATOR_CUH__

#include <cuda_runtime.h>
#include <stdexcept>

template <typename T>
struct HostAllocator
{
    const char *target = "host";
    void allocate(size_t size, T **p_start, T **p_end)
    {
        *p_start = (T *)malloc(size * sizeof(T));
        *p_end = (*p_start) + size;
    }
    void deallocate(T **p_start, T **p_end)
    {
        if (*p_start)
        {
            free(*p_start);
            *p_start = nullptr;
            *p_end = nullptr;
        }
    }
    void copy_from_host(T *dst, const T *src, size_t size)
    {
        memcpy(dst, src, size * sizeof(T));
    }
    void copy_from_device(T *dst, const T *src, size_t size)
    {
        auto result = cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy from device to host memory");
        }
        // fprintf(stdout, "copy %lu elements from device to host\n", size);
    }
};

template <typename T>
struct DeviceAllocator
{
    const char *target = "cuda";
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
    void copy_from_host(T *dst, const T *src, size_t size)
    {
        auto result = cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy from host to device memory");
        }
        // fprintf(stdout, "copy %lu elements from host to device\n", size);
    }
    void copy_from_device(T *dst, const T *src, size_t size)
    {
        throw std::runtime_error("NotImplemented");
    }
};

#endif /* __ALLOCATOR_CUH__ */
