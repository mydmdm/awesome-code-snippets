import os
import sys
import subprocess
import re


BUILD_DIR, BIN_DIR = "build", "bin"
BIN_DIR = os.path.join(BUILD_DIR, BIN_DIR)
os.makedirs(BIN_DIR, exist_ok=True)


def nvcc_cmds():
    cmds = ["nvcc"]
    cmds.extend(["-I/usr/local/cuda/include"])  # include
    cmds.extend(["-lstdc++", "-L/usr/local/cuda/lib64", "-lcublas", "-lcudart"])  # library
    cmds.extend(["-std=c++11 -O3 -m64 -w --resource-usage --ptxas-options=-v -lineinfo -Xcompiler -fPIC -Wno-deprecated-gpu-targets"])  # flags

    # check gpu arch
    gpu_tester = os.path.join(BIN_DIR, 'gpu-info')
    if not os.path.isfile(gpu_tester):
        print(f"compiling for {gpu_tester}")
        subprocess.run(cmds + ["device_info.cu", "-o", gpu_tester])
    result = subprocess.run([gpu_tester], stdout=subprocess.PIPE).stdout.splitlines()[0].decode("utf-8")
    arch = re.search("\(cm_[\d]+\)", result).group()[1:-1]
    cmds.append(f"-arch={arch}")
    return cmds


# matmul compiling
src, nvcc = ["matmul.cu"], nvcc_cmds()
comp_cmds, run_cmds = [], []
for tile in [16]:
    targ = os.path.join(BIN_DIR, f"matmul")
    comp_cmds.append(nvcc + ["-o", targ] + src)
    run_cmds.append([targ])

subprocess.run(comp_cmds[0])
subprocess.run(run_cmds[0])
