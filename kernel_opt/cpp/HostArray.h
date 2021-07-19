#ifndef __HOSTARRAY_H__
#define __HOSTARRAY_H__

template <typename T>
struct HostAllocator
{
    void allocate(size_t size, T **p_start, T **p_end)
    {
        *p_start = (T *)malloc(size * sizeof(T));
        *p_end = (*p_start) + size;
    }
    void free(T **p_start, T **p_end)
    {
        if (*p_start)
        {
            free(*p_start);
            *p_start = nullptr;
            *p_end = nullptr;
        }
    }
};

template <typename T, class Allc>
class ArrayBase
{
public:
    ArrayBase(size_t size)
    {
        Allc.allocate(size, &_start, &_end);
    }

    ~ArrayBase()
    {
        Allc.free(&_start, &_end);
    }

    inline size_t size() const
    {
        return _end - _start;
    }

    inline const T *data(size_t offset = 0u) const
    {
        return _start + offset;
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
};

template <typename T>
class HostArray : public ArrayBase<T, HostAllocator<T>()>
{
    using ArrayBase<T, HostAllocator<T>()>::ArrayBase;
};

#endif /* __HOSTARRAY_H__ */
