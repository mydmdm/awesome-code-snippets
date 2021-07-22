#include "Tensors.h"
#include "compute.cuh"
#include "compute.h"

#define M 1024
#define N 1024
#define K 512

#define D float

#define TILE 16

#define num_repeat 100

#define HOSTALLOC PinnedHostAllocator

#define TEST(cu_kernel)                                                         \
    do                                                                          \
    {                                                                           \
        auto now = get_now();                                                   \
        range(i, num_repeat)                                                    \
        {                                                                       \
            cu_kernel<<<blocks, threads>>>(d_a._start, d_b._start, d_c._start); \
        }                                                                       \
        cudaDeviceSynchronize();                                                \
        auto t = time_difference_ns(now);                                       \
        fprintf(stdout, "%s, %e\n", #cu_kernel, (double)t / num_repeat);        \
        Matrix<D, M, N> c(hoa, &d_c);                                           \
        double max_err;                                                         \
        auto num_err = count_error(c0, c, &max_err);                            \
        if (num_err)                                                            \
        {                                                                       \
            fprintf(stdout, "max error is %e\n", max_err);                      \
            fprintf(stdout, "number of error is %lu\n", num_err);               \
        }                                                                       \
    } while (0)

/* compute C = A*B, A is (M,K), B is (K,N), and C is (M,N)
*/
int main(int argc, char *argv[])
{

    fprintf(stdout, "matmul test with shape (M,N,K)=(%d,%d,%d)\n", M, N, K);

    auto hoa = PinnedHostAllocator<D>(); // host memory allocator
    auto dva = DeviceAllocator<D>();
    Matrix<D, M, K> a(hoa);
    Matrix<D, K, N> b(hoa);
    Matrix<D, M, N> c0(hoa);

    randint<D>(a, 0.0, 100.0);
    randint<D>(b, 0.0, 100.0);
    // set_const<D>(a, 1.0);
    // set_const<D>(b, 1.0);

    if (1)
    {
        auto now = get_now();
        matmul<D, M, N, K>(a, b, c0);
        auto t = time_difference_ns(now);
        fprintf(stdout, "naive_cpu, %e\n", (double)t);
        // is_const<D>(c0, K);
    }

    Matrix<D, M, K> d_a(dva, &a);
    Matrix<D, K, N> d_b(dva, &b);
    Matrix<D, M, N> d_c(dva);
    dim3 threads(TILE, TILE);
    dim3 blocks(iceil(N, TILE), iceil(M, TILE));
    fprintf(stdout, "launing kernel blocks=(%u,%u), threads=(%u,%u)\n", blocks.x, blocks.y, threads.x, threads.y);

    TEST(cu_matmul_naive<D COMMA M COMMA N COMMA K>);

    TEST(cu_matmul_tiled<D COMMA M COMMA N COMMA K COMMA TILE>);

    TEST(cu_matmul_tiled_t<D COMMA M COMMA N COMMA K COMMA TILE>);
}