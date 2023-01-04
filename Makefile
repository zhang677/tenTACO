all: tensor-eval-gpu
NVCC = /home/eva_share/opt/cuda-11.1/bin/nvcc
CXX = icpc
OPTS = -O3 -DNDEBUG -march=native -mtune=native -ffast-math -fopenmp
CC_FLAGS = -Xcompiler -fPIC -shared --std=c++14
CUDA_INCLUDE_DIR = /home/eva_share/opt/cuda-11.1/include 
CUDA_LIBRARY_DIR = /home/eva_share/opt/cuda-11.1/lib64 
SPLATT_INCLUDE_DIR = /home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/splatt/include
SPLATT_LIBRARY_DIR = /home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/splatt/build/Linux-x86_64/lib
EIGEN_INCLUDE_DIR = /home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/eigen-3.4.0


tensor-eval-gpu: tensor_main_gpu.o Makefile
	$(NVCC) -o tensor-eval-gpu -I . -DGPU -L ${SPLATT_LIBRARY_DIR} -lcuda -lcudart -L ${CUDA_LIBRARY_DIR} -lcusparse tensor_main_gpu.o gpu_kernels.cu -lpthread -lm -ldl -lsplatt -liomp5 -lblas -llapack

tensor_main_gpu.o: tensor_main.cpp Makefile
	$(NVCC) -o tensor_main_gpu.o $(CC_FLAGS) -I . -I ${EIGEN_INCLUDE_DIR} -I ${SPLATT_INCLUDE_DIR} -lcuda -lcudart -I ${CUDA_INCLUDE_DIR} tensor_main.cpp

.PHONY: clean run

clean:
	rm -f tensor-eval-gpu
	rm -f *.o

run:
	./run_gpu.sh