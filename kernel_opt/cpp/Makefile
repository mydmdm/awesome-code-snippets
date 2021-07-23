CXX_COMP = g++
CXX_FLAGS_DEV = -c -O3 -std=c++11 -fPIC -D_FORCE_INLINES -g
CXX_FLAGS_OPT = -c -O3 -std=c++11 -fPIC -D_FORCE_INLINES
CXX_FLAGS = $(CXX_FLAGS_OPT)
CXX_INC =
CXX_LIB = -lstdc++

CUDA_COMP = nvcc
CUDA_HOST = /usr/bin/g++
CUDA_INC = -I/usr/local/cuda/include
CUDA_LIB = -L/usr/local/cuda/lib64 -lcublas -lcudart
CUDA_COMMON_FLAGS = -ccbin $(CUDA_HOST) -std=c++11 -arch=$(CUDA_ARCH) -m64 -w --resource-usage --ptxas-options=-v -Xcompiler -fPIC 
CUDA_FLAGS_DEV = ${CUDA_COMMON_FLAGS} -O0 -g -G
CUDA_FLAGS_OPT = ${CUDA_COMMON_FLAGS} -O3
CUDA_FLAGS_ADV = ${CUDA_COMMON_FLAGS} -O3 -lineinfo 
CUDA_FLAGS = $(CUDA_FLAGS_ADV)

LINK_FLAGS = -fPIC

.PHONY: clean arch matmul
# run matmul test
gpu-info: device_info.cu
	$(CUDA_COMP) -o $@.exe $<

arch: gpu-info
	$(eval CUDA_ARCH := sm_$(shell ./gpu-info.exe | head -1 | sed -n 's/.*cm_\([0-9]\+\).*/\1/p'))
	echo ${CUDA_ARCH}

matmul: matmul.cu arch
	$(CUDA_COMP) $(CUDA_FLAGS) $(CXX_INC) $(CUDA_INC) -o $@.exe $<

clean:
	rm -f *.out *.x *.a *.so *.o *.mod *.modmic *.ptx *.log
	rm -f *.exe