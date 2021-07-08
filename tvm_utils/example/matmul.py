import tvm
import tvm.testing
from tvm import te
import numpy
import timeit
from tvm_utils.utils import *


def create_compute(M: int, N: int, K: int):
    """return the compute of matrix multiplication of A and B, the shape is
    (M, K) x (K, N)
    """
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    return (A, B, C)


def create_cuda_schedule(stages, block_size, tx, ty, tk):
    """create basic cuda scheduling with blocking and tiling
    """
    A, B, C = stages
    s = te.create_schedule(C.op)
    # Create caches
    A_shared = s.cache_read(A, "shared", [C])
    A_local = s.cache_read(A_shared, "local", [C])
    B_shared = s.cache_read(B, "shared", [C])
    B_local = s.cache_read(B_shared, "local", [C])
    C_local = s.cache_write(C, "local")
    # Split each axis into block axis, thread axis, and inner axis
    x, y = s[C].op.axis
    xb, xo, xi = split_from_inner(s[C], x, (block_size, tx))
    yb, yo, yi = split_from_inner(s[C], y, (block_size, ty))
    s[C].reorder(xb, yb, xo, yo, xi, yi)
    # Note that we bind yb to blockIdx.x instead of blockIdx.y
    bind_thread(s[C], (yb, xb, yo, xo),
                ("blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y"))
    # Schedule C_local
    s[C_local].compute_at(s[C], yo)
    yi, xi = s[C_local].op.axis
    k, = s[C_local].op.reduce_axis
    ko, ki = s[C_local].split(k, tk)
    s[C_local].reorder(ko, ki, yi, xi)
    # Optimize read caches of A and B with cooperative fetching

    def optimize_read_cache(shared, local):
        s[shared].compute_at(s[C_local], ko)
        s[local].compute_at(s[C_local], ki)
        y, x = s[shared].op.axis
        # Note that we must split into block_size parts to reuse
        # the previous axis threads
        yo, yi = s[shared].split(y, nparts=block_size)
        xo, xi = s[shared].split(x, nparts=block_size)
        s[shared].reorder(yo, xo, yi, xi)
        bind_thread(s[shared], (yo, xo), ("threadIdx.y", "threadIdx.x"))

    optimize_read_cache(A_shared, A_local)
    optimize_read_cache(B_shared, B_local)
    return s


def cuda_impl(M: int, N: int, K: int, schedule, dtype: str = "float32", target: str = "cuda", repeat: int = 100, name: str = None):
    A, B, C = create_compute(M, N, K)
    s = schedule((A, B, C))

    dev = tvm.device(target, 0)
    name = name or f'{target}_impl'
    func = tvm.build(s, [A, B, C], target=target, name=name)
    assert func

    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)
    answer = numpy.dot(a.numpy(), b.numpy())

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, number=repeat)
    cuda_running_time = evaluator(a, b, c).mean
    print(f"Baseline: {cuda_running_time}")
    return func, s, cuda_running_time


def baseline_numpy(M: int, N: int, K: int, dtype: str = "float32", repeat: int = 100):
    """return the latency of numpy implementation
    """
    np_setup_cmds = f'''
import numpy
M, N, K = {M}, {N}, {K}
dtype = "{dtype}"
a = numpy.random.rand(M, K).astype(dtype)
b = numpy.random.rand(K, N).astype(dtype)
'''
    # print(np_setup_cmds)
    np_runing_time = timeit.timeit(
        setup=np_setup_cmds,
        stmt="answer = numpy.dot(a, b)",
        number=repeat,
    )
    np_runing_time /= repeat
    print(f"Numpy running time: {np_runing_time}")
    return np_runing_time
