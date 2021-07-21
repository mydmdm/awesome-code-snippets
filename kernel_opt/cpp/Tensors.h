#ifndef __TENSORS_H__
#define __TENSORS_H__

#include "Allocator.cuh"
#include "utils.h"

template <typename T, size_t SIZE>
struct Array : public Array_<T>
{
    static const size_t _size = SIZE;

    Array(Allocator_<T> allc, Array_<T> *src = nullptr)
    {
        allc.allocate(SIZE, this);
        if (src)
        {
            assert_eq(SIZE, len(src), "SizeMismatch");
            check_cuda(copy_memory<T>(this, src));
        }
    }

    ~Array()
    {
        switch (this->_type)
        {
        case MemoryType::HostNaive:
            NaiveHostAllocator<T>().deallocate(this);
            return;
        case MemoryType::HostPinned:
            PinnedHostAllocator<T>().deallocate(this);
            return;
        case MemoryType::Device:
            DeviceAllocator<T>().deallocate(this);
            return;
        }
        assert_true(false, "NotImplemented");
    }
};

template <typename T, size_t ROWS, size_t COLS>
struct Matrix : public Array<T, ROWS * COLS>
{
    using Array<T, ROWS * COLS>::Array;
    static const size_t _rows = ROWS;
    static const size_t _cols = COLS;

    inline T &at(size_t i, size_t j)
    {
        return this->_start[i * COLS + j];
    }

    inline const T &at(size_t i, size_t j) const
    {
        return this->_start[i * COLS + j];
    }
};

#endif /* __TENSORS_H__ */
