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

template <typename T, size_t SIZE, class ALLC>
struct Array : public Array_<T>
{
    static const size_t _size = SIZE;

    Array()
    {
        ALLC().allocate(SIZE, this);
    }

    Array(Array_<T> &src)
    {
        assert_eq(SIZE, len(&src), "SizeMismatch");
        ALLC().allocate(SIZE, this);
        check_cuda(copy_memory<T>(this, &src));
    }

    ~Array()
    {
        ALLC().deallocate(this);
    }
};

#endif /* __TENSORS_H__ */
