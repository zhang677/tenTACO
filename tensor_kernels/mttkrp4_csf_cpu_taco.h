#ifndef MTTKRP4_CPU_TACO_H
#define MTTKRP4_CPU_TACO_H

#include "ds.h"
#include "timers.h"

void mttkrp4_cpu_taco(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D, taco_tensor_t *E) {
TIME_COLD(
  splatt_idx_t A1_dimension = (splatt_idx_t)(A->dimensions[0]);
  splatt_idx_t A2_dimension = (splatt_idx_t)(A->dimensions[1]);
  double* restrict A_vals = (double*)(A->vals);
  splatt_idx_t B1_dimension = (splatt_idx_t)(B->dimensions[0]);
  splatt_idx_t* restrict B2_pos = (splatt_idx_t*)(B->indices[1][0]);
  splatt_idx_t* restrict B2_crd = (splatt_idx_t*)(B->indices[1][1]);
  splatt_idx_t* restrict B3_pos = (splatt_idx_t*)(B->indices[2][0]);
  splatt_idx_t* restrict B3_crd = (splatt_idx_t*)(B->indices[2][1]);
  splatt_idx_t* restrict B4_pos = (splatt_idx_t*)(B->indices[3][0]);
  splatt_idx_t* restrict B4_crd = (splatt_idx_t*)(B->indices[3][1]);
  double* restrict B_vals = (double*)(B->vals);
  splatt_idx_t C1_dimension = (splatt_idx_t)(C->dimensions[0]);
  splatt_idx_t C2_dimension = (splatt_idx_t)(C->dimensions[1]);
  double* restrict C_vals = (double*)(C->vals);
  splatt_idx_t D1_dimension = (splatt_idx_t)(D->dimensions[0]);
  splatt_idx_t D2_dimension = (splatt_idx_t)(D->dimensions[1]);
  double* restrict D_vals = (double*)(D->vals);
  splatt_idx_t E1_dimension = (splatt_idx_t)(E->dimensions[0]);
  splatt_idx_t E2_dimension = (splatt_idx_t)(E->dimensions[1]);
  double* restrict E_vals = (double*)(E->vals);

  _Pragma("omp parallel for schedule(static)")
  for (splatt_idx_t pA = 0; pA < (A1_dimension * A2_dimension); pA++) {
    A_vals[pA] = 0.0;
  }

  _Pragma("omp parallel for schedule(dynamic, 1)")
  for (splatt_idx_t i1 = 0; i1 < ((B1_dimension + 63) / 64); i1++) {
    for (splatt_idx_t i2 = 0; i2 < 64; i2++) {
      splatt_idx_t i = i1 * 64 + i2;
      if (i >= B1_dimension)
        continue;

      for (splatt_idx_t kB = B2_pos[i]; kB < B2_pos[(i + 1)]; kB++) {
        splatt_idx_t k = B2_crd[kB];
        for (splatt_idx_t lB = B3_pos[kB]; lB < B3_pos[(kB + 1)]; lB++) {
          splatt_idx_t l = B3_crd[lB];
          for (splatt_idx_t mB = B4_pos[lB]; mB < B4_pos[(lB + 1)]; mB++) {
            splatt_idx_t m = B4_crd[mB];
            for (splatt_idx_t j = 0; j < E2_dimension; j++) {
              splatt_idx_t jA = i * A2_dimension + j;
              splatt_idx_t jC = k * C2_dimension + j;
              splatt_idx_t jD = l * D2_dimension + j;
              splatt_idx_t jE = m * E2_dimension + j;
              A_vals[jA] = A_vals[jA] + ((B_vals[mB] * C_vals[jC]) * D_vals[jD]) * E_vals[jE];
            }
          }
        }
      }
    }
  }
);
}

#endif
