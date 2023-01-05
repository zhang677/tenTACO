#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H
#include "taco_tensor_t.h"
float get_gpu_timer_result();
void mttkrp_gpu_taco(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void ttv_csf_gpu_taco(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C);
void mttkrp_32_eb_sr_512_4_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_64_32_manual(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_64_eb_pr_256_32_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_32_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_32_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_32_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_32_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_64_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_64_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_64_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_64_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_128_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_128_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_128_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_128_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_256_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_256_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_256_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_256_256_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_64_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_64_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_64_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_64_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_128_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_128_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_128_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_128_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_256_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_256_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_256_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_256_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_512_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_512_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_512_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_pr_512_512_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_4_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_4_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_4_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_4_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_8_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_8_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_8_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_8_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_16_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_16_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_16_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_16_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_32_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_32_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_32_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_256_32_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_4_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_4_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_4_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_4_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_8_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_8_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_8_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_8_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_16_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_16_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_16_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_16_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_32_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_32_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_32_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
void mttkrp_32_eb_sr_512_32_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);
#endif