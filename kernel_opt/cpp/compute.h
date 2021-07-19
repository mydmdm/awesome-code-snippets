#ifndef __COMPUTE_H__
#define __COMPUTE_H__

#include "utils.h"

template <typename T>
T dot_product(const T *a, const T *b, size_t size, size_t step_a = 1u, size_t step_b = 1u)
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

template <class MAT = MatR<float>>
void matmul_naive(MAT a, MAT b, MAT c)
{
    assert_eq(a.cols, b.rows, "InputShapeCheck");
    range(i, a.rows)
    {
        range(j, b.cols)
        {
            *c.at(i, j) = 0;
            range(k, a.cols)
            {
                *c.at(i, j) += (*a.at(i, k)) * (*b.at(k, j));
            }
        }
    }
}

#endif /* __COMPUTE_H__ */
