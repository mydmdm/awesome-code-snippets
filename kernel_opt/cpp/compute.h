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
size_t count_error(Array_<T> &a, Array_<T> &b, double *max_err = nullptr, double epison = 1e-6)
{
    size_t num = 0;
    T err = 0;
    range(i, len<T>(&a))
    {
        auto tmp = std::abs(a._start[i] - b._start[i]);
        num += tmp > epison;
        err = std::max(err, tmp);
    }
    if (max_err)
    {
        *max_err = err;
    }
    return num;
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

template <typename T>
void randint(Array_<T> &arr, int vmin, int vmax)
{
    for (auto p = arr._start; p != arr._end; ++p)
    {
        *p = (rand() % (vmax - vmin)) + vmin;
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

template <typename T, size_t C_ROWS, size_t C_COLS, size_t WIDTH>
void matmul(const Matrix<T, C_ROWS, WIDTH> &a, const Matrix<T, WIDTH, C_COLS> &b, Matrix<T, C_ROWS, C_COLS> &c)
{
    range(x, C_ROWS)
    {
        range(y, C_COLS)
        {
            c.at(x, y) = 0;
            range(z, WIDTH)
            {
                c.at(x, y) += a.at(x, z) * b.at(z, y);
            }
        }
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
