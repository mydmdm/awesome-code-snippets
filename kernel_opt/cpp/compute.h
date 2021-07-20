#ifndef __COMPUTE_H__
#define __COMPUTE_H__

#include "Tensors.h"

#include <random>

template <typename T>
void set_const(Array_<T> &arr, T val)
{
    for (auto p = arr._start; p != arr._end; ++p)
    {
        *p = val;
    }
}

template <typename T>
bool is_const(Array_<T> &arr, T val)
{
    for (auto p = arr._start; p != arr._end; ++p)
    {
        if (*p != val)
        {
            return false;
        }
    }
    return true;
}

template <typename T>
void randn(Array_<T> &arr, T mean, T std)
{
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(mean, std);
    for (auto p = arr._start; p != arr._end; ++p)
    {
        *p = distribution(generator);
    }
}

template <typename T, size_t size>
void vector_add(const T *a, const T *b, T *c)
{
    range(i, size)
    {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
T dot_product(const T *a, const T *b, size_t size, size_t step_a, size_t step_b)
{
    T c = 0;
    range(i, size)
    {
        c += (*a) * (*b);
        a += step_a;
        b += step_b;
    }
    return c;
}

#endif /* __COMPUTE_H__ */
