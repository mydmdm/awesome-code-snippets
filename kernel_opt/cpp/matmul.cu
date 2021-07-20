#include "Tensors.h"
#include "compute.h"

#ifndef D
#define D float
#endif

int main(int argc, char *argv[])
{
#if !(defined(M) && defined(N) && defined(K))
    assert_eq(argc, 4, "InputArgs");
    auto M = atoi(argv[1]);
    auto N = atoi(argv[2]);
    auto K = atoi(argv[3]);
#else
    assert_eq(argc, 1, "InputArgs");
#endif
    fprintf(stdout, "matmul test with shape (M,N,K)=(%d,%d,%d)\n", M, N, K);

    HostArray<DT> a(M * K), b(K * N), c0(M * N);
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