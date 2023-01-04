#!/bin/bash
#MATRICES_DIR=/home/nfs_data/zhanggh/mytaco/learn-taco/tensors/
#TENSOR_NAME=part_names.txt
MATRICES_DIR=/home/nfs_data/datasets/sparse_ten
TENSOR_NAME=3mode.txt
RESULTS_DIR=./tensor_result
export LD_LIBRARY_PATH="/home/eva_share/opt/cuda-11.1/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/splatt/build/Linux-x86_64/lib:$LD_LIBRARY_PATH"
CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 ./taco-eval.py $MATRICES_DIR mttkrp_csf_gpu $RESULTS_DIR $2 $3 $TENSOR_NAME



