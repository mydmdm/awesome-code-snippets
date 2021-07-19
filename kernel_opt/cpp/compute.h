#ifndef __COMPUTE_H__
#define __COMPUTE_H__

#include <cstddef>

template <typename T>
T dot_product(const T *a, const T *b, size_t size)
{
    T c = 0;
    for (auto i = 0u; i != size; ++i)
    {
        c += a[i] * b[i];
    }
    return c;
}

#endif /* __COMPUTE_H__ */
