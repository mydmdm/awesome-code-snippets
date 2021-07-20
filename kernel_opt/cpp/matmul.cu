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

/* compute C = A*B^T, A is (M,K), B is (N,K), and C is (M,N)
*/
int main(int argc, char *argv[])
{

    fprintf(stdout, "matmul test with shape (M,N,K)=(%d,%d,%d)\n", M, N, K);

    auto hoa = PinnedHostAllocator<D>(); // host memory allocator
    auto dva = DeviceAllocator<D>();
    Matrix<D, M, K> a(hoa);
    Matrix<D, N, K> b(hoa);
    Matrix<D, M, N> c0(hoa);

    // randn<D>(a, 0.0, 1.0);
    // randn<D>(b, 0.0, 1.0);
    set_const<D>(a, 1.0);
    set_const<D>(b, 1.0);

    Matrix<D, M, K> d_a(dva, &a);
    Matrix<D, N, K> d_b(dva, &b);
    Matrix<D, M, N> d_c(dva);

    if (1)
    {
        auto now = get_now();
        range(x, M)
        {
            range(y, N)
            {
                c0._start[x * N + y] = 0;
                range(z, K)
                {
                    c0._start[x * N + y] += a._start[x * K + z] * b._start[y * K + z];
                }
            }
        }
        auto t = time_difference_ns(now);
        fprintf(stdout, "naive_cpu, %lu\n", t);
        is_const<D>(c0, K);
    }
}