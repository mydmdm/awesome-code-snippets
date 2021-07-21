#include "Tensors.h"
#include "compute.h"

#if !defined(M) || !defined(N) || !defined(K)
#define M 1024
#define N 1024
#define K 512
#endif

#ifndef D
#define D float
#endif

// number of threads per block
#ifndef TPB
#define TPB 32, 32
#endif

#ifndef num_repeat
#define num_repeat 100
#endif

#define HOSTALLOC PinnedHostAllocator

const dim3 shape(N, M);

__global__ void cu_matmul_naive(const D *__restrict__ a, const D *__restrict__ b, D *__restrict__ c)
{
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        c[row * N + col] = 0;
        range(z, K)
        {
            c[row * N + col] += a[row * K + z] * b[z * K + col];
        }
    }
}

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

    // randn<D>(a, 0.0, 1.0);
    // randn<D>(b, 0.0, 1.0);
    set_const<D>(a, 1.0);
    set_const<D>(b, 1.0);

    Matrix<D, M, K> d_a(dva, &a);
    Matrix<D, K, N> d_b(dva, &b);
    Matrix<D, M, N> d_c(dva);

    if (1)
    {
        auto now = get_now();
        range(x, M)
        {
            range(y, N)
            {
                c0.at(x, y) = 0;
                range(z, K)
                {
                    c0.at(x, y) += a.at(x, z) * b.at(z, y);
                }
            }
        }
        auto t = time_difference_ns(now);
        fprintf(stdout, "naive_cpu, %e\n", (double)t);
        is_const<D>(c0, K);
    }

    if (1)
    {
        dim3 threads(TPB);
        dim3 blocks(iceil(shape.x, threads.x), iceil(shape.y, threads.y));
        auto now = get_now();
        range(i, num_repeat)
        {
            cu_matmul_naive<<<blocks, threads>>>(d_a._start, d_b._start, d_c._start);
        }
        cudaDeviceSynchronize();
        auto t = time_difference_ns(now);
        fprintf(stdout, "cu_matmul_naive, %e\n", (double)t / num_repeat);
        Matrix<D, M, N> c(hoa, &d_c);
        assert_true(memcmp(c0._start, c._start, M * N * sizeof(D)) == 0, "ComputeWrong");
    }
}