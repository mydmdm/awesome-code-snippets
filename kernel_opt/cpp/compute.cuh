#ifndef __COMPUTE_CUH__
#define __COMPUTE_CUH__

#include "assert.h"

template <typename T, size_t C_ROWS, size_t C_COLS, size_t WIDTH>
__global__ void cu_matmul_naive(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c)
{
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < C_ROWS && col < C_COLS)
    {
        c[row * C_COLS + col] = 0;
        range(z, WIDTH)
        {
            // auto idx_a = row * WIDTH + z;
            // auto idx_b = z * C_COLS + col;
            // assert(idx_a < C_ROWS * WIDTH);
            // assert(idx_b < WIDTH * C_COLS);
            c[row * C_COLS + col] += a[row * WIDTH + z] * b[z * C_COLS + col];
        }
    }
}

/* a rectangular (non-square) tile may cause lots of problems
 * let's restrict the TILE to be a square
*/
template <typename T, size_t C_ROWS, size_t C_COLS, size_t WIDTH, size_t TILE>
__global__ void cu_matmul_tiled(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c)
{
    __shared__ T a_s[TILE][TILE];
    __shared__ T b_s[TILE][TILE];

    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < C_ROWS && col < C_COLS)
    {
        T value = 0;
        range(t, WIDTH / TILE)
        {
            a_s[threadIdx.y][threadIdx.x] = a[row * WIDTH + (t * TILE + threadIdx.x)];
            b_s[threadIdx.y][threadIdx.x] = b[(t * TILE + threadIdx.y) * C_COLS + col];
            __syncthreads();
            range(z, TILE)
            {
                value += a_s[threadIdx.y][z] * b_s[z][threadIdx.x];
            }
            __syncthreads();
        }
        c[row * C_COLS + col] = value;
    }
}

template <typename T, size_t C_ROWS, size_t C_COLS, size_t WIDTH, size_t TILE>
__global__ void cu_matmul_tiled_t(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c)
{
    __shared__ T a_s[TILE][TILE];
    __shared__ T b_s[TILE][TILE];

    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < C_ROWS && col < C_COLS)
    {
        T value = 0;
        range(t, WIDTH / TILE)
        {
            a_s[threadIdx.y][threadIdx.x] = a[row * WIDTH + (t * TILE + threadIdx.x)];
            b_s[threadIdx.x][threadIdx.y] = b[(t * TILE + threadIdx.y) * C_COLS + col];
            __syncthreads();
            range(z, TILE)
            {
                value += a_s[threadIdx.y][z] * b_s[threadIdx.x][z];
            }
            __syncthreads();
        }
        c[row * C_COLS + col] = value;
    }
}

#endif /* __COMPUTE_CUH__ */
