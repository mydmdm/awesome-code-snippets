import pycuda.tools as ct
# import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as ga

import numpy as np
import math
import sys
import os
import argparse
from jinja2 import Template
import torch
import torch.testing
import traceback


# verify device availability
assert torch.cuda.is_available(), "GPU device is not available"
__device__ = torch.device('cuda')
__gpu_info__ = ct.DeviceData()
print(__gpu_info__.__dict__)
# print(torch.cuda.get_device_properties(torch.cuda.current_device()))

# src = Template('''
# __global__ void test()
# {
#         printf("<debug> hello, test @ block[%d,%d,%d], thread[%d,%d,%d]\\n", 
#             blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z); 
# }
# ''').render()
# print(src)
# func = SourceModule(src).get_function('test')
# func(block=(32,32,1), grid=(1,1,1))
# torch.cuda.synchronize()
# # time.sleep(10)
# # sys.exit(0)

class TesterBase:

    def __init__(self, repetitions: int=100, warmup: int=10, cmp_kw: dict=None) -> None:
        self.repetitions = repetitions
        self.warmup = warmup
        self.cmp_kw = cmp_kw or {}

    def test_function(self, func, f_args: list, f_kw: dict, checker: callable= None):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        #GPU-WARM-UP
        for i in range(self.warmup):
            tmp = func(*f_args, **f_kw)
            torch.cuda.synchronize() # WAIT FOR GPU SYNC
            if checker is not None and i==0: # verify correctness on the first time
                checker(tmp)

        if self.repetitions ==0:
            return 0,0
        # MEASURE PERFORMANCE
        timings=np.zeros((self.repetitions,1))
        with torch.no_grad():
            for rep in range(self.repetitions):
                starter.record()
                _ = func(*f_args, **f_kw)
                ender.record()
                torch.cuda.synchronize() # WAIT FOR GPU SYNC
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / self.repetitions
        std_syn = np.std(timings)
        return mean_syn, std_syn


class MatmulTest(TesterBase):

    def __init__(self, m: int, n: int, k: int, dtype: str = 'float', tester_cfg: dict=None) -> None:
        tester_cfg = tester_cfg or {}
        super().__init__(**tester_cfg)

        self.m, self.n, self.k, self.dtype = m, n, k, dtype
        self.flops = 2.0 * m * n * k

        self.results = []
        self.prepare_data()

    def add_record(self, name:str, lat_ms: float, **kwargs):
        # print('received one record for', name)
        record = dict(
            name=name, lat_ms=lat_ms, gflops=self.flops * 1e-9 / (lat_ms or 1.0)
        )
        record.update(kwargs)
        self.results.append(record)

    def prepare_data(self):
        a0 = np.random.randn(self.m, self.k).astype(np.float32)
        b0 = np.random.randn(self.k, self.n).astype(np.float32)
        self.c0 = torch.from_numpy(a0 @ b0).cuda()
        assert self.c0.shape[0] == self.m and self.c0.shape[1] == self.n
        self.c = np.zeros((self.m, self.n), dtype=np.float32)
        self.d_a = ga.to_gpu(a0)
        self.d_b = ga.to_gpu(b0)
        self.d_c = ga.to_gpu(self.c)

        a = torch.from_numpy(a0).cuda()
        b = torch.from_numpy(b0).cuda()
        t = self.test_function(torch.matmul, [a, b], dict(), 
            checker = lambda r: torch.testing.assert_close(r, self.c0, **self.cmp_kw)
        )
        self.add_record('torch', t[0])

    def profile_kernel(self, name: str, func: callable, blocksPerGrid, threadsPerBlock):
        def __func():
            self.d_c.fill(0) # reset to zero
            return func(self.d_a, self.d_b, self.d_c, block=threadsPerBlock, grid=blocksPerGrid)

        t = self.test_function(
            __func, [], dict(),
            checker = lambda r: torch.testing.assert_close(torch.from_numpy(self.d_c.get()).cuda(), self.c0, **self.cmp_kw)
            )
        r = torch.from_numpy(self.d_c.get()).cuda()
        if torch.max(torch.abs(r - self.c0)) > 1e-2:
            print(r, self.c0)
        self.add_record(name, t[0], threads=threadsPerBlock, blocks=blocksPerGrid)      
        return t

    # def profile(self, ki: KernelImplement, threads: tuple, blocks: tuple = None):
    #     threads = threads + (1,) if len(threads) == 2 else threads
    #     blocks = blocks or tuple([math.ceil(x / threads[i]) for i, x in enumerate([self.n, self.m, 1])])
    #     assert threads[2] == 1 and threads[0] * threads[1] <= self.gpu_prop.max_threads
    #     t = self.test_function(
    #         ki.func_call, [self.d_a, self.d_b, self.d_c], dict(block=threads, grid=blocks),
    #         checker = lambda r: torch.testing.assert_close(self.d_c, self.c0, **self.cmp_kw)
    #         )
    #     self.add_record(ki.name, t[0], threads=threads, blocks=blocks)
    #     return
    #     self.d_c.fill(0)
    #     e_start, e_end = cuda.Event(), cuda.Event()
    #     e_start.record()
    #     ki.func_call(self.d_a, self.d_b, self.d_c, block=threads, grid=blocks)
    #     e_end.record()
    #     e_end.synchronize()
    #     cuda.memcpy_dtoh(self.c, self.d_c.ptr)
    #     secs = e_start.time_till(e_end)*1e-3
    #     gflops = self.flops * 1e-9 / secs
    #     err = np.max(np.abs(self.c0 - self.c))
    #     return dict(kernel=ki.name, max_err=err, latency=secs, gflops=gflops, threads=threads, blocks=blocks)


class KernelBase:
    __default_name__: str=None

    def __init__(self, func: str=None, comp_opts: list = None) -> None:
        self.name = func or self.__default_name__
        assert self.name is not None, f"name={func}, default={self.__default_name__}"
        with open(os.path.join('templates', self.name + ".h.j2")) as fn:
            self.src_template = Template(fn.read())
        self.compile_opts = comp_opts or []

    def run(self, tester: MatmulTest, *args, **kwargs):
        blocksPerGrid, threadsPerBlock, tpl_args = self.generate_args(*args, **kwargs)
        print(f'profiling kernel {self.name}<<<{blocksPerGrid},{threadsPerBlock}>>>')
        try:
            src = self.src_template.render(**tpl_args)
            func_call = SourceModule(src, options=self.compile_opts).get_function(self.name)
            return tester.profile_kernel(self.name, func_call, blocksPerGrid, threadsPerBlock)

        except Exception as e:
            # print('Error raised:', e)
            traceback.print_exc()
            self.print_source_code(tpl_args)
            sys.exit(-1)
        # tester.profile_kernel(self.name, self.func_call, blocksPerGrid, threadsPerBlock)
          

    def print_source_code(self, args: dict, lineno: bool = True):
        src_code = self.src_template.render(**args)
        if lineno:
            lines = src_code.splitlines()
            for i, line in enumerate(lines):
                print(f"{i+1:3d}\t{line}")
        else:
            print(src_code)

    def generate_args(self, *args, **kw):
        raise NotImplementedError


class KernelV0(KernelBase):
    __default_name__: str = 'cu_matmul_naive'

    def generate_args(self, M: int, N: int, K: int, dtype: str='float'):
        tpl_args = dict(T=dtype, C_ROWS=M, C_COLS=N, WIDTH=K)
        thrds = int(math.sqrt(__gpu_info__.max_threads))
        threadsPerBlock = (thrds, thrds, 1)
        blocksPerGrid = (math.ceil(N/thrds), math.ceil(M/thrds), 1)
        return blocksPerGrid, threadsPerBlock, tpl_args


class KernelV1(KernelBase):
    __default_name__: str = 'cu_matmul_tiled'

    def generate_args(self, M: int, N: int, K: int, tile:int=None, dtype: str='float'):
        if tile:
            thrds = tile
        else:
            thrds = int(math.sqrt(__gpu_info__.max_threads))
        tpl_args = dict(T=dtype, C_ROWS=M, C_COLS=N, WIDTH=K, TILE=thrds)
        threadsPerBlock = (thrds, thrds, 1)
        blocksPerGrid = (math.ceil(N/thrds), math.ceil(M/thrds), 1)
        return blocksPerGrid, threadsPerBlock, tpl_args


class KernelV3(KernelBase):
    __default_name__: str = 'cu_matmul_tiled_3'

    def generate_args(self, M: int, N: int, K: int, 
        tiles: tuple, threads: tuple,
        dtype: str='float'):
        bm, bn, bk = tiles
        tpl_args = dict(T=dtype, M=M, N=N, K=K, 
            BM=bm, BN=bn, BK=bk)
        threadsPerBlock = threads
        blocksPerGrid = (math.ceil(M/bm), math.ceil(N/bn), 1)
        return blocksPerGrid, threadsPerBlock, tpl_args


if __name__ == '__main__':
    from tabulate import tabulate
    parser = argparse.ArgumentParser('profiling C(mxn) = A(mxk) * B(kxn)')
    parser.add_argument('-m', type=int, default=2000, help='number of rows of A and C')
    parser.add_argument('-n', type=int, default=2000, help='number of columns of C')
    parser.add_argument('-k', type=int, default=2000, help='number of rows of B')
    parser.add_argument('-d', '--debug-mode', action='store_true', help='debug mode')
    args = parser.parse_args()

    compile_opts = ['-Wno-deprecated-gpu-targets']    

    tc = MatmulTest(args.m, args.n, args.k, tester_cfg={'cmp_kw':{'atol': 5e-4, 'rtol':1e-3}})

    if args.debug_mode:
        tc.warmup, tc.repetitions = 1, 0
        compile_opts.append('-DDEBUG')

    # naive 
    ki = KernelV0()
    ki.run(tc, args.m, args.n, args.k)

    # naive tiled
    ki = KernelV1()
    ki.run(tc, args.m, args.n, args.k)

    # 
    ki = KernelV3(comp_opts=compile_opts)
    ki.run(tc, args.m, args.n, args.k, (32,32,32), (32,32,1))


    # threads = tuple([int(math.sqrt(tc.gpu_prop.max_threads))] * 2)  # max number of thread per block
    # kernels = [
    #     KernelImplement('cu_matmul_naive', tc.tpl_args),
    #     KernelImplement('cu_matmul_tiled', tc.tpl_args, {"TILE": 32}),
    # ]
    # for tk in [32, 64, 128, 160]:
    #     kernels.append(KernelImplement(
    #         'cu_matmul_tiled_2', tc.tpl_args,
    #         {"TM": threads[1], "TN": threads[0], "TK": tk}
    #     ))
    # for ki in kernels:
    #     tc.profile(ki, threads)

    print(tabulate(tc.results, headers="keys"))

    # result = [tc.profile(ki, threads) for ki in kernels]
    # print(tabulate(result, headers="keys"))
