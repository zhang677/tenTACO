#include "gpu_kernels.cuh"
#include "gpu_library.h"

#include "tensor_kernels/mttkrp3_csf_gpu_taco.h"
#include "tensor_kernels/ttv_csf_gpu_taco.h"
#include "tensor_kernels/mttkrp3_csf_ebpr.h"

GPUTimer gpu_timer;

float get_gpu_timer_result() {
    return gpu_timer.get_result();
}