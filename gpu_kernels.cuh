#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H
#include "taco_tensor_t.h"
float get_gpu_timer_result();
void mttkrp_gpu_taco(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void ttv_csf_gpu_taco(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C);
#endif