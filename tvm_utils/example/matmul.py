import tvm
import tvm.testing
from tvm import te
import numpy
import timeit


def create_compute(M: int, N: int, K: int):
    """return the compute of matrix multiplication of A and B, the shape is
    (M, K) x (K, N)
    """
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    return A, B, C


def baseline_numpy(M: int, N: int, K: int, dtype: str = "float32", repeat: int = 100):
    """return the latency of numpy implementation
    """
    np_runing_time = timeit.timeit(
        setup=f'''import numpy as np
M, N, K = {M}, {N}, {K}
a = numpy.random.rand(M, K).astype({dtype})
b = numpy.random.rand(K, N).astype({dtype})
        ''',
        stmt="answer = numpy.dot(a, b)",
        number=repeat,
    )
    print("Numpy running time: %f" % (np_runing_time / np_repeat))
    return np_runing_time
