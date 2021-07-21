#include "Tensors.h"
#include "compute.h"

#define M 1024
#define N 1024
#define K 512

#define D float

#define TILE_R 32
#define TILE_C 32

#define num_repeat 100

#define HOSTALLOC PinnedHostAllocator

__global__ void cu_matmul_naive(const D *__restrict__ a, const D *__restrict__ b, D *__restrict__ c)
{
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        c[row * N + col] = 0;
        range(z, K)
        {
            c[row * N + col] += a[row * K + z] * b[z * N + col];
        }
    }
}

__global__ void cu_matmul_tiled(const D *__restrict__ a, const D *__restrict__ b, D *__restrict__ c)
{
    __shared__ D a_s[TILE_R][TILE_C];
    __shared__ D b_s[TILE_R][TILE_C];

    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N)
    {
        D value = 0;
        range(t, K / TILE_C)
        {
            a_s[threadIdx.y][threadIdx.x] = a[row * K + (t * TILE_R + threadIdx.x)];
            b_s[threadIdx.y][threadIdx.x] = b[(t * TILE_C + threadIdx.y) * N + col];
            __syncthreads();
            range(z, TILE_C)
            {
                value += a_s[threadIdx.y][z] * b_s[z][threadIdx.x];
            }
            __syncthreads();
        }
        c[row * N + col] = value;
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

    randint<D>(a, 0.0, 100.0);
    randint<D>(b, 0.0, 100.0);
    // set_const<D>(a, 1.0);
    // set_const<D>(b, 1.0);

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
        // is_const<D>(c0, K);
    }

    if (1)
    {
        dim3 threads(TILE_C, TILE_R);
        dim3 blocks(iceil(N, threads.x), iceil(M, threads.y));
        auto now = get_now();
        range(i, num_repeat)
        {
            cu_matmul_naive<<<blocks, threads>>>(d_a._start, d_b._start, d_c._start);
        }
        cudaDeviceSynchronize();
        auto t = time_difference_ns(now);
        fprintf(stdout, "cu_matmul_naive, %e\n", (double)t / num_repeat);
        Matrix<D, M, N> c(hoa, &d_c);
        fprintf(stdout, "max error is %e\n", max_error(c0, c));
        // assert_true(memcmp(c0._start, c._start, M * N * sizeof(D)) == 0, "ComputeWrong");
    }

    if (1)
    {
        dim3 threads(TILE_C, TILE_R);
        dim3 blocks(iceil(N, threads.x), iceil(M, threads.y));
        auto now = get_now();
        range(i, num_repeat)
        {
            cu_matmul_tiled<<<blocks, threads>>>(d_a._start, d_b._start, d_c._start);
        }
        cudaDeviceSynchronize();
        auto t = time_difference_ns(now);
        fprintf(stdout, "cu_matmul_tiled, %e\n", (double)t / num_repeat);
        Matrix<D, M, N> c(hoa, &d_c);
        fprintf(stdout, "max error is %e\n", max_error(c0, c));
        // assert_true(memcmp(c0._start, c._start, M * N * sizeof(D)) == 0, "ComputeWrong");
    }
}