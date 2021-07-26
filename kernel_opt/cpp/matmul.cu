#include "Tensors.h"
#include "compute.cuh"
#include "compute.h"
#include "utils_cublas.cuh"

#include "matmul_def.h"

struct TestContext
{
    FILE *_fn;
    PinnedHostAllocator<D> a_host;
    DeviceAllocator<D> a_dev;
    cublasHandle_t handler;

    Matrix<D, M, K> a;
    Matrix<D, K, N> b;
    Matrix<D, M, N> c0; // ground truth
    // Matrix<D, M, N> c;

    Matrix<D, M, K> d_a;
    Matrix<D, K, N> d_b;
    // Matrix<D, M, N> d_c;

    double flops = 2.0 * M * N * K; // number of MAC (Multiply-Accumulate) operations

    TestContext(const char *fname)
        : a(a_host), b(a_host), c0(a_host),
          d_a(a_dev), d_b(a_dev)
    {
        if (fname)
        {
            _fn = fopen(fname, "a+");
        }
        checkCublasStatus(cublasCreate(&handler));

        randint<D>(a, 0.0, 100.0);
        randint<D>(b, 0.0, 100.0);
        // set_const<D>(a, 1.0);
        // set_const<D>(b, 1.0);
        copy_memory<D>(&d_a, &a);
        copy_memory<D>(&d_b, &b);

        fprintf(stdout, "generate matrix muliplication test with shape (%lu, %lu, %lu), Gflop = %.2f\n", M, N, K, flops * 1e-9);
        if (flops >= 1e9)
        {
            fprintf(stdout, "too large computation for cpu reference, use cublas for correctness check\n");
            fprintf(stdout, "use a small (M,N,K) for cpu correctness check\n");
            Matrix<D, M, N> d_c(a_dev);
            cublas_matmal<D, M, N, K>(handler, d_a._start, d_b._start, d_c._start);
            // checkCudaStatus(cudaDeviceSynchronize());
            copy_memory<D>(&c0, &d_c);
        }
        else
        {
            auto name = "naive_cpu";
            auto now = get_now();
            matmul<D, M, N, K>(a, b, c0);
            auto t = time_difference_ns(now);
            fprintf(stdout, "%s, latency(ns), %e, Gflop/s, %.2f\n", name, (double)t, flops / t);
        }
    }

    ~TestContext()
    {
        checkCublasStatus(cublasDestroy(handler));
        if (_fn)
        {
            fclose(_fn);
        }
    }

    void verify(const char *name, double t, Matrix<D, M, N> *d_c)
    {
        fprintf(stdout, "name=%s, latency=%.9f, Gflop/s=%.3f\n", t * 1e-9, flops / t);
        if (_fn)
        {
            fprintf(_fn, "name=%s, latency=%.9f, Gflop/s=%.3f\n", t * 1e-9, flops / t);
        }
        Matrix<D, M, N> c(a_host, d_c);
        double max_err;
        auto num_err = count_error(c0, c, &max_err);
        if (num_err)
        {
            fprintf(stdout, "max error is %e\n", max_err);
            fprintf(stdout, "number of error is %lu\n", num_err);
        }
    }

    void test_kernel(const char *name, void (*kernel)(const D *, const D *, D *), dim3 blocks, dim3 threads)
    {
        Matrix<D, M, N> d_c(a_dev);
        auto now = get_now();
        range(i, num_repeat)
        {
            kernel<<<blocks, threads>>>(d_a._start, d_b._start, d_c._start);
        }
        checkCudaStatus(cudaDeviceSynchronize());
        auto t = time_difference_ns(now);
        verify(name, t, &d_c);
    }

    void test_cublas()
    {
        Matrix<D, M, N> d_c(a_dev);
        auto now = get_now();
        range(i, num_repeat)
        {
            cublas_matmal<D, M, N, K>(handler, d_a._start, d_b._start, d_c._start);
        }
        checkCudaStatus(cudaDeviceSynchronize());
        auto t = time_difference_ns(now);
        verify("cublas", t, &d_c);
    }
};

/* compute C = A*B, A is (M,K), B is (K,N), and C is (M,N)
*/
int main(int argc, char *argv[])
{
    auto prop = print_device_properties();
    fprintf(stdout, "matmul test with shape (M,N,K)=(%d,%d,%d)\n", M, N, K);
    int algo = 0;
    std::string fname = "result.matmul.txt";

    TestContext tc(fname.c_str());

    if (algo == (int)AlgoSel::all || algo == (int)AlgoSel::cublas)
    {
        tc.test_cublas();
    }

    dim3 threads(TILE, TILE);
    dim3 blocks(iceil(N, TILE), iceil(M, TILE));
    fprintf(stdout, "launing kernel blocks=(%u,%u), threads=(%u,%u)\n", blocks.x, blocks.y, threads.x, threads.y);

    if (algo == (int)AlgoSel::all || algo == (int)AlgoSel::naive)
    {
        tc.test_kernel("cu_matmul_naive", cu_matmul_naive<D, M, N, K>, blocks, threads);
    }
    if (algo == (int)AlgoSel::all || algo == (int)AlgoSel::tiled)
    {
        tc.test_kernel("cu_matmul_tiled", cu_matmul_tiled<D, M, N, K, TILE>, blocks, threads);
    }
}