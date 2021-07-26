import os
import sys
import subprocess
import re
from typing import DefaultDict
import argparse


BUILD_DIR, BIN_DIR = "build", "bin"
BIN_DIR = os.path.join(BUILD_DIR, BIN_DIR)
os.makedirs(BIN_DIR, exist_ok=True)


def nvcc_cmds():
    cmds = ["nvcc"]
    cmds.extend(["-ccbin", "/usr/bin/g++"])
    cmds.extend(["-I/usr/local/cuda/include"])  # include
    cmds.extend(["-lstdc++", "-L/usr/local/cuda/lib64", "-lcublas", "-lcudart"])  # library
    cmds.extend("-std=c++11 -O3 -m64 -w --resource-usage --ptxas-options=-v -lineinfo -Xcompiler -fPIC -Wno-deprecated-gpu-targets".split())  # flags

    # check gpu arch
    gpu_tester = os.path.join(BIN_DIR, 'gpu-info')
    if not os.path.isfile(gpu_tester):
        print(f"compiling for {gpu_tester}")
        subprocess.run(cmds + ["device_info.cu", "-o", gpu_tester])
    result = subprocess.run([gpu_tester], stdout=subprocess.PIPE).stdout.splitlines()[0].decode("utf-8")
    arch = re.search("\(cm_[\d]+\)", result).group()[4:-1]
    cmds.append(f"-arch=sm_{arch}")
    return cmds


class TestBase:

    def __init__(self) -> None:
        self.name = None
        self.definitions = DefaultDict(dict)
        self.comp_cmds = []
        self.run_cmds = []

    def generate_definition_header(self):
        fname = self.name + "_def.h"
        guard = '__' + fname.replace('.', '_').upper() + '__'
        with open(fname, "w") as fn:
            fn.write(f"#ifndef {guard}\n")
            fn.write(f"#define {guard}\n\n")
            for k, v in self.definitions['macros'].items():
                fn.write(f"#ifndef {k}\n#define {k} {v}\n#endif\n\n")
            for ename, evals in self.definitions['enums'].items():
                fn.write(f"enum class {ename} {{\n\t")
                fn.write(',\n\t'.join([f'{k}={v}' for k, v in evals.items()]))
                fn.write(f"\n}};\n\n")
            fn.write(f"#endif\n")

    def run(self):
        for cc in self.comp_cmds:
            print(' '.join(cc))
            subprocess.run(cc)
        for cc in self.run_cmds:
            print(cc)
            subprocess.run(cc)


class MatmulTest(TestBase):

    def __init__(self) -> None:
        super().__init__()
        self.name = "matmul"
        algos = dict(all=0, cublas=1, naive=2, tiled=3)
        self.definitions = {
            "macros": dict(D='float', M=2000, N=2000, K=2000, TILE=16, num_repeat=100),
            "enums": {"AlgoSel": algos},
        }
        nvcc = nvcc_cmds()
        for tile in [16, 32]:
            targ = os.path.join(BIN_DIR, f"matmul_t{tile}")
            rd = [f"-DTILE={tile}"]
            # rd = []
            self.comp_cmds.append(nvcc + rd + ["-o", targ, self.name+'.cu'])
            self.run_cmds.append([targ, algos['tiled']])


if __name__ == '__main__':
    test = MatmulTest()
    test.generate_definition_header()
    test.run()
