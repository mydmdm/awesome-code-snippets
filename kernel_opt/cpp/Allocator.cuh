#ifndef __ALLOCATOR_CUH__
#define __ALLOCATOR_CUH__

#include "utils.cuh"
#include "utils.h"

template <typename T>
inline int len(Array_<T> *obj)
{
    return obj->_end - obj->_start;
}

template <typename T>
inline bool is_device(Array_<T> *obj)
{
    return obj->_type == MemoryType::Device;
}

using fn_allocate_t = cudaError_t (*)(void **, size_t);
using fn_free_t = cudaError_t (*)(void *);

template <typename T>
struct Allocator_
{
    fn_allocate_t _allocate_fn;
    fn_free_t _free_fn;
    MemoryType _type;
    Allocator_(fn_allocate_t f1, fn_free_t f2, MemoryType t) : _allocate_fn(f1), _free_fn(f2), _type(t) {}

    void allocate(size_t size, Array_<T> *obj)
    {
        auto result = this->_allocate_fn((void **)&obj->_start, size * sizeof(T));
        checkCudaStatus(result);
        obj->_end = obj->_start + size;
        obj->_type = this->_type;
    }
    void deallocate(Array_<T> *obj)
    {
        if (obj->_start)
        {
            auto result = this->_free_fn(obj->_start);
            checkCudaStatus(result);
            obj->_start = obj->_end = nullptr;
        }
    }
};

cudaError_t malloc_naive(void **ptr, size_t bytes)
{
    *ptr = malloc(bytes);
    return cudaSuccess;
}

cudaError_t free_naive(void *ptr)
{
    free(ptr);
    return cudaSuccess;
}

template <typename T>
struct PageableHostAllocator : public Allocator_<T>
{
    PageableHostAllocator() : Allocator_<T>(malloc_naive, free_naive, MemoryType::HostPageable) {}
};

template <typename T>
struct PinnedHostAllocator : public Allocator_<T>
{
    PinnedHostAllocator() : Allocator_<T>(cudaMallocHost, cudaFreeHost, MemoryType::HostPinned) {}
};

template <typename T>
struct DeviceAllocator : public Allocator_<T>
{
    DeviceAllocator() : Allocator_<T>(cudaMalloc, cudaFree, MemoryType::Device) {}
};

template <typename T>
cudaError_t copy_memory(Array_<T> *dst, Array_<T> *src)
{
    auto bytes = len(src) * sizeof(T);
    if (!is_device(dst) && !is_device(src)) // from host to host
    {
        memcpy(dst->_start, src->_start, bytes);
        return cudaSuccess;
    }
    if (is_device(dst) && !is_device(src)) // from host to device
    {
        return cudaMemcpy(dst->_start, src->_start, bytes, cudaMemcpyHostToDevice);
    }
    if (!is_device(dst) && is_device(src)) // from device to host
    {
        return cudaMemcpy(dst->_start, src->_start, bytes, cudaMemcpyDeviceToHost);
    }
    assert_true(false, "NotImplemented");
}

#endif /* __ALLOCATOR_CUH__ */
