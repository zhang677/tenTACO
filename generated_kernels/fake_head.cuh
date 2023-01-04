#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H
#include "taco_tensor_t.h"
float get_gpu_timer_result();
void mttkrp_gpu_taco(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void ttv_csf_gpu_taco(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C);
void mttkrp_32_eb_sr_512_4_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_64_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void spmm_32_eb_pr_256_64_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_64_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_64_eb_pr_256_32_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_256_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_256_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_64_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
#endif
