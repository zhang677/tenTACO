#ifndef MTTKRP3_CPU_TACO_WORKSPACE_H
#define MTTKRP3_CPU_TACO_WORKSPACE_H

#include "ds.h"
#include "timers.h"
#include <cstdlib>

void mttkrp3_cpu_taco_workspace(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
TIME_COLD(
  splatt_idx_t A1_dimension = (splatt_idx_t)(A->dimensions[0]);
  splatt_idx_t A2_dimension = (splatt_idx_t)(A->dimensions[1]);
  double* restrict A_vals = (double*)(A->vals);
  splatt_idx_t B1_dimension = (splatt_idx_t)(B->dimensions[0]);
  splatt_idx_t* restrict B2_pos = (splatt_idx_t*)(B->indices[1][0]);
  splatt_idx_t* restrict B2_crd = (splatt_idx_t*)(B->indices[1][1]);
  splatt_idx_t* restrict B3_pos = (splatt_idx_t*)(B->indices[2][0]);
  splatt_idx_t* restrict B3_crd = (splatt_idx_t*)(B->indices[2][1]);
  double* restrict B_vals = (double*)(B->vals);
  splatt_idx_t C1_dimension = (splatt_idx_t)(C->dimensions[0]);
  splatt_idx_t C2_dimension = (splatt_idx_t)(C->dimensions[1]);
  double* restrict C_vals = (double*)(C->vals);
  splatt_idx_t D1_dimension = (splatt_idx_t)(D->dimensions[0]);
  splatt_idx_t D2_dimension = (splatt_idx_t)(D->dimensions[1]);
  double* restrict D_vals = (double*)(D->vals);

  _Pragma("omp parallel for schedule(static)")
  for (splatt_idx_t pA = 0; pA < (A1_dimension * A2_dimension); pA++) {
    A_vals[pA] = 0.0;
  }

  _Pragma("omp parallel for schedule(dynamic, 1)")
  for (splatt_idx_t i1 = 0; i1 < ((B1_dimension + 63) / 64); i1++) {
    double* restrict precomputed = 0;
    precomputed = (double*)malloc(sizeof(double) * C2_dimension);
    for (splatt_idx_t pprecomputed = 0; pprecomputed < C2_dimension; pprecomputed++) {
      precomputed[pprecomputed] = 0.0;
    }
    for (splatt_idx_t i2 = 0; i2 < 64; i2++) {
      splatt_idx_t i = i1 * 64 + i2;
      if (i >= B1_dimension)
        continue;

      for (splatt_idx_t kB = B2_pos[i]; kB < B2_pos[(i + 1)]; kB++) {
        splatt_idx_t k = B2_crd[kB];
        for (splatt_idx_t lB = B3_pos[kB]; lB < B3_pos[(kB + 1)]; lB++) {
          splatt_idx_t l = B3_crd[lB];
          for (splatt_idx_t j = 0; j < C2_dimension; j++) {
            splatt_idx_t jD = l * D2_dimension + j;
            precomputed[j] = precomputed[j] + B_vals[lB] * D_vals[jD];
          }
        }
        for (splatt_idx_t j = 0; j < C2_dimension; j++) {
          splatt_idx_t jA = i * A2_dimension + j;
          splatt_idx_t jC = k * C2_dimension + j;
          A_vals[jA] = A_vals[jA] + precomputed[j] * C_vals[jC];
          precomputed[j] = 0;
        }
      }
    }
    free(precomputed);
  }
);
}

#endif
