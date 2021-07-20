#include "Tensors.h"
#include "compute.h"

#if !defined(M) || !defined(N) || !defined(K)
#define M 1024
#define N 512
#define K 256
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

#define HOSTALLOC PinnedHostAllocator

int main(int argc, char *argv[])
{

    fprintf(stdout, "matmul test with shape (M,N,K)=(%d,%d,%d)\n", M, N, K);

    Matrix<D, M, K, HOSTALLOC<D>> a;
    Matrix<D, N, K, HOSTALLOC<D>> a;

    a.randn(0.0, 1.0);
    b.randn(0.0, 1.0);
    if (1)
    {
        auto now = get_now();
        matmul_naive(a.matr(M, K), b.matr(K, N), c0.matr(M, N));
        auto t = time_difference_ns(now);
        fprintf(stdout, "naive, %lu\n", t);
    }
    if (1)
    {
        HostArray<DT> c(M * N);
        auto now = get_now();
        range(i, M)
        {
            range(j, N)
            {
                c.data[i * N + j] = dot_product(a.data + i * K, b.data + j, K, 1u, N);
            }
        }
        auto t = time_difference_ns(now);
        fprintf(stdout, "naive_dot, %lu\n", t);
    }
}