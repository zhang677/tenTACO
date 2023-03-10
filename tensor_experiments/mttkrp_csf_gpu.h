#ifndef MTTKRP_CSF_GPU_H
#define MTTKRP_CSF_GPU_H

#include "ds.h"
#include "timers.h"
#include <iostream>

#include "gpu_kernels.cuh"

void mttkrp_ref(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
  int32_t A1_dimension = (int32_t)(A->dimensions[0]);
  int32_t A2_dimension = (int32_t)(A->dimensions[1]);
  float* restrict A_vals = (float*)(A->vals);
  int32_t B1_dimension = (int32_t)(B->dimensions[0]);
  int32_t* restrict B2_pos = (int32_t*)(B->indices[1][0]);
  int32_t* restrict B2_crd = (int32_t*)(B->indices[1][1]);
  int32_t* restrict B3_pos = (int32_t*)(B->indices[2][0]);
  int32_t* restrict B3_crd = (int32_t*)(B->indices[2][1]);
  float* restrict B_vals = (float*)(B->vals);
  int32_t C1_dimension = (int32_t)(C->dimensions[0]);
  int32_t C2_dimension = (int32_t)(C->dimensions[1]);
  float* restrict C_vals = (float*)(C->vals);
  int32_t D1_dimension = (int32_t)(D->dimensions[0]);
  int32_t D2_dimension = (int32_t)(D->dimensions[1]);
  float* restrict D_vals = (float*)(D->vals);

  _Pragma("omp parallel for schedule(static)")
  for (int32_t pA = 0; pA < (A1_dimension * A2_dimension); pA++) {
    A_vals[pA] = 0.0;
  }
   _Pragma("omp parallel for schedule(dynamic, 16)")
  for (int64_t i = 0; i < B1_dimension; i++) {
    for (int64_t pB2 = B2_pos[i]; pB2 < B2_pos[(i + 1)]; pB2++) {
      int64_t k = B2_crd[pB2];
      for (int64_t pB3 = B3_pos[pB2]; pB3 < B3_pos[(pB2 + 1)]; pB3++) {
        int64_t l = B3_crd[pB3];
        for (int64_t j = 0; j < D2_dimension; j++) {
          int64_t pA2 = i * A2_dimension + j;
          int64_t pC2 = k * C2_dimension + j;
          int64_t pD2 = l * D2_dimension + j;

		      B_vals[pB3]=1; C_vals[pC2]=1; D_vals[pD2]=1;

          A_vals[pA2] = A_vals[pA2] + B_vals[pB3] * C_vals[pC2] * D_vals[pD2];
        }
      }
    }
  }
}

void mttkrp_csf_gpu(splatt_csf* B_splatt, const std::string& tensor_name, const bool do_verify, int num_cols) {
  if (B_splatt->nmodes != 3) {
    std::cout << "skipping as not 3 modes" << std::endl;
    return;
  }
  taco_tensor_t B_taco = to_taco_tensor(B_splatt);
  int J = num_cols;
  
  splatt_kruskal factor_matrices = gen_factor_matrices(J, B_splatt);
  std::vector<taco_tensor_t> mats(B_splatt->nmodes);
  for (int i = 0; i < B_splatt->nmodes; ++i) {
    mats[i] = to_taco_tensor(factor_matrices, B_splatt, i);
  }

  if (do_verify) {
	std::vector<taco_tensor_t> mats_ref(B_splatt->nmodes);
	splatt_kruskal factor_matrices_ref = gen_factor_matrices(J, B_splatt);
  	for (int i = 0; i < B_splatt->nmodes; ++i) {
	    mats_ref[i] = to_taco_tensor(factor_matrices_ref, B_splatt, i);
	}

	mttkrp_ref(&mats_ref[0], &B_taco, &mats[1], &mats[2]);
  compress_top_level(B_taco); // compress top level
  /*
	mttkrp_32_eb_pr_256_32_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_32_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_32_16(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_32_32(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_64_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_64_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_64_16(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_64_32(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_128_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_128_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_128_16(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_128_32(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_256_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_256_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_256_16(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_256_256_32(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  */
  /*
  mttkrp_32_eb_pr_512_64_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_64_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_64_16(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_64_32(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_128_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_128_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_128_16(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_128_32(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_256_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_256_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_256_16(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_256_32(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_512_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_512_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_512_16(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_pr_512_512_32(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  */
  mttkrp_32_eb_sr_256_4_1(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_4_2(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_4_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_4_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_8_1(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_8_2(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_8_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_8_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_16_1(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_16_2(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_16_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_16_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_32_1(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_32_2(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_32_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_256_32_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_4_1(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_4_2(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_4_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_4_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_8_1(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_8_2(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_8_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_8_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_16_1(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_16_2(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_16_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_16_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_32_1(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_32_2(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_32_4(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;
  mttkrp_32_eb_sr_512_32_8(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << "taco CPU vs taco GPU: " << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;

  exit(0);
  }

  compress_top_level(B_taco); // compress top level

  const int trials = 25;
  /*
  RUN_GPU( mttkrp_gpu_taco(&mats[0], &B_taco, &mats[1], &mats[2]);, trials, "mttkrp", "gpu", "csf", "taco", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_32_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune0", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_32_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune1", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_32_16(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune2", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_32_32(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune3", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_64_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune4", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_64_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune5", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_64_16(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune6", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_64_32(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune7", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_128_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune8", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_128_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune9", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_128_16(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune10", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_128_32(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune11", tensor_name);
  */
  /*
  RUN_GPU( mttkrp_32_eb_pr_256_256_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune12", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_256_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune13", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_256_16(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune14", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_256_256_32(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune15", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_64_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune16", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_64_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune17", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_64_16(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune18", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_64_32(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune19", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_128_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune20", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_128_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune21", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_128_16(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune22", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_128_32(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune23", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_256_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune24", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_256_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune25", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_256_16(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune26", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_256_32(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune27", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_512_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune28", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_512_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune29", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_512_16(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune30", tensor_name);
  RUN_GPU( mttkrp_32_eb_pr_512_512_32(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "pr", "tune31", tensor_name);
  */
  RUN_GPU( mttkrp_32_eb_sr_256_4_1(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune0", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_4_2(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune1", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_4_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune2", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_4_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune3", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_8_1(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune4", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_8_2(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune5", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_8_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune6", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_8_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune7", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_16_1(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune8", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_16_2(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune9", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_16_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune10", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_16_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune11", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_32_1(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune12", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_32_2(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune13", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_32_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune14", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_256_32_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune15", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_4_1(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune16", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_4_2(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune17", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_4_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune18", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_4_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune19", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_8_1(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune20", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_8_2(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune21", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_8_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune22", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_8_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune23", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_16_1(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune24", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_16_2(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune25", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_16_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune26", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_16_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune27", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_32_1(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune28", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_32_2(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune29", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_32_4(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune30", tensor_name);
  RUN_GPU( mttkrp_32_eb_sr_512_32_8(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,"mttkrp", "eb", "sr", "tune31", tensor_name);
}

#endif
