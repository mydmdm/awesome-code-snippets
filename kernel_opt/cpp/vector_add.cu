#include "Allocator.cuh"
#include "Tensors.h"
#include "compute.h"
#include "utils.h"

#include <stdio.h>

#ifndef N
#define N (1 << 12)
#endif

#ifndef D
#define D float
#endif

// number of threads per block
#ifndef TPB
#define TPB 256
#endif

#ifndef num_repeat
#define num_repeat 100
#endif

/* use template parameter to transfer const parameter and use __restrict__ could help nvcc to optimize
*/
template <typename T, size_t SIZE>
__global__ void cu_vec_add_naive(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < SIZE)
        c[tid] = a[tid] + b[tid];
}

int main()
{
    auto hoa = PinnedHostAllocator<D>(); // host memory allocator
    auto dva = DeviceAllocator<D>();
    Array<D, N> a(hoa), b(hoa), c0(hoa);
    // randn<D>(a, 0.0, 1.0);
    // randn<D>(b, 0.0, 1.0);
    set_const<D>(a, 1.0);
    set_const<D>(b, 1.0);

    Array<D, N> d_a(dva, &a), d_b(dva, &b), d_c(dva);

    if (1)
    {
        auto now = get_now();
        range(i, num_repeat)
        {
            vector_add<D, N>(a._start, b._start, c0._start);
        }
        auto t = time_difference_ns(now);
        fprintf(stdout, "naive_cpu, %lu\n", t / num_repeat);
        is_const<D>(c0, 2.0);
    }

    if (1)
    {
        auto now = get_now();
        range(i, num_repeat)
        {
            cu_vec_add_naive<D, N><<<iceil(N, TPB), TPB>>>(d_a._start, d_b._start, d_c._start);
        }
        cudaDeviceSynchronize();
        auto t = time_difference_ns(now);
        fprintf(stdout, "cu_vec_add_naive, %lu\n", t / num_repeat);
        Array<D, N> c(hoa, &d_c);
        assert_true(memcmp(c0._start, c._start, N * sizeof(D)) == 0, "ComputeWrong");
    }
}
