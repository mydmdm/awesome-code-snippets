#ifndef __HOSTARRAY_H__
#define __HOSTARRAY_H__

#include <random>

template <typename T>
struct HostAllocator
{
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
};

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

template <typename T, class Allc>
class ArrayBase
{
public:
    T *data;

    ArrayBase(size_t size)
    {
        _allocator.allocate(size, &_start, &_end);
        data = _start;
    }

    ~ArrayBase()
    {
        _allocator.deallocate(&_start, &_end);
    }

    inline size_t size() const
    {
        return _end - _start;
    }

    inline MatR<T> matr(size_t rows, size_t cols)
    {
        return MatR<T>(rows, cols, _start);
    }

    void randn(T mean, T std)
    {
        std::default_random_engine generator;
        std::normal_distribution<T> distribution(mean, std);
        for (auto p = _start; p != _end; ++p)
        {
            *p = distribution(generator);
        }
    }

protected:
    T *_start{nullptr};
    T *_end{nullptr};
    Allc _allocator;
};

template <typename T>
class HostArray : public ArrayBase<T, HostAllocator<T>>
{
    using ArrayBase<T, HostAllocator<T>>::ArrayBase;
};

#endif /* __HOSTARRAY_H__ */
