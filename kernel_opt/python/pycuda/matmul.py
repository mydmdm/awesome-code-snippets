from typing import overload
import pycuda.tools as ct
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as ga
import numpy
import math
import os
import argparse
from jinja2 import Template


class KernelImplement:

    def __init__(self, func: str, c_args: dict = None, e_args: dict = None, comp_opts: list = None) -> None:
        self.src_template = None
        with open(os.path.join('templates', func + ".h.j2")) as fn:
            self.src_template = fn.read()
        tpl_args = dict(c_args or {})
        tpl_args.update(e_args or {})
        self.src_code = Template(self.src_template).render(**tpl_args)
        try:
            comp_opts = comp_opts or ['-Wno-deprecated-gpu-targets']
            self.func_call = SourceModule(self.src_code, options=comp_opts).get_function(func)
        except:
            self.print_source_code()
            assert False

    def print_source_code(self, lineno: bool = True):
        if lineno:
            lines = self.src_code.splitlines()
            for i, line in enumerate(lines):
                print(f"{i+1:3d}\t{line}")
        else:
            print(self.src_code)


class MatmulTest:

    def __init__(self, m: int, n: int, k: int, dtype: str = 'float') -> None:
        self.m, self.n, self.k, self.dtype = m, n, k, dtype
        self.flops = 2.0 * m * n * k
        self.tpl_args = dict(T=dtype, C_ROWS=m, C_COLS=n, WIDTH=k)
        self.gpu_prop = ct.DeviceData()
        self.test_case()

    def test_case(self):
        a = numpy.random.randn(self.m, self.k).astype(numpy.float32)
        b = numpy.random.randn(self.k, self.n).astype(numpy.float32)
        self.c0 = a @ b
        assert self.c0.shape[0] == self.m and self.c0.shape[1] == self.n
        self.c = numpy.zeros_like(self.c0).astype(numpy.float32)
        self.d_a = ga.to_gpu(a)
        self.d_b = ga.to_gpu(b)
        self.d_c = ga.to_gpu(self.c)

    def profile(self, ki: KernelImplement, threads: tuple, blocks: tuple = None):
        threads = threads + (1,) if len(threads) == 2 else threads
        blocks = blocks or tuple([math.ceil(x / threads[i]) for i, x in enumerate([self.n, self.m, 1])])
        print(threads, blocks)
        self.d_c.fill(0)
        e_start, e_end = cuda.Event(), cuda.Event()
        e_start.record()
        ki.func_call(self.d_a, self.d_b, self.d_c, block=threads, grid=blocks)
        e_end.record()
        e_end.synchronize()
        cuda.memcpy_dtoh(self.c, self.d_c.ptr)
        secs = e_start.time_till(e_end)*1e-3
        gflops = self.flops * 1e-9 / secs
        err = numpy.max(numpy.abs(self.c0 - self.c))
        return dict(max_err=err, latency=secs, gflops=gflops, threads=threads, blocks=blocks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('profiling C(mxn) = A(mxk) * B(kxn)')
    parser.add_argument('-m', type=int, default=2000, help='number of rows of A and C')
    parser.add_argument('-n', type=int, default=2000, help='number of columns of C')
    parser.add_argument('-k', type=int, default=2000, help='number of rows of B')
    args = parser.parse_args()

    tc = MatmulTest(args.m, args.n, args.k)
    ki = KernelImplement('cu_matmul_naive', tc.tpl_args)
    result = tc.profile(ki, (32, 32))
    print(result)
