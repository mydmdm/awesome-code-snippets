#ifndef __TENSORS_H__
#define __TENSORS_H__

#include "Allocator.cuh"
#include "utils.h"

template <typename T>
struct MatR // row-majored matrix
{
    MatR(size_t r, size_t c, T *p) : rows(r), cols(c), data(p) {}
    size_t rows, cols;
    T *data;
    inline T *at(size_t i, size_t j) const
    {
        return data + i * cols + j;
    }
};

template <typename T>
struct Array_
{
    T *_start{nullptr};
    T *_end{nullptr};
};

template <typename T, size_t SIZE, class ALLC>
struct Array : public Array_<T>
{
    static const size_t _size = SIZE;
    ALLC _allocator;

    Array()
    {
        _allocator.allocate(SIZE, &this->_start, &this->_end);
    }

    Array(Array<T, SIZE, HostAllocator<T>> &src)
    {
        _allocator.allocate(SIZE, &this->_start, &this->_end);
        _allocator.copy_from_host(this->_start, src._start, SIZE);
    }

    Array(Array<T, SIZE, DeviceAllocator<T>> &src)
    {
        _allocator.allocate(SIZE, &this->_start, &this->_end);
        _allocator.copy_from_device(this->_start, src._start, SIZE);
    }

    ~Array()
    {
        _allocator.deallocate(&this->_start, &this->_end);
    }
};

#endif /* __TENSORS_H__ */
