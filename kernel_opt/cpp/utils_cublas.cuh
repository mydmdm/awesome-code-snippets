#ifndef __UTILS_CUBLAS_CUH__
#define __UTILS_CUBLAS_CUH__

#include "Tensors.h"
#include "utils.h"
#include <cublas_v2.h>
#include <stdexcept>

inline void checkCublasStatus(cublasStatus_t result, const char *msg = nullptr)
{
    if (result != CUBLAS_STATUS_SUCCESS)
    {
        if (msg)
        {
            throw std::runtime_error(msg);
        }
        else
        {
            throw std::runtime_error("operation failed");
        }
    }
}

/* input / output are row-majored matrices
 */
template <typename T, size_t C_ROWS, size_t C_COLS, size_t WIDTH>
void cublas_matmal(cublasHandle_t handler, const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
{
    const T alpha = 1;
    const T beta = 0;
    // Matrix<T, C_COLS, C_ROWS> d_c_t(DeviceAllocator<T>());

    checkCublasStatus(cublasSgemm(handler, CUBLAS_OP_N, CUBLAS_OP_N, C_COLS, C_ROWS, WIDTH, &alpha, B, C_COLS, A, WIDTH, &beta, C, C_COLS));
}

#endif /* __UTILS_CUBLAS_CUH__ */
