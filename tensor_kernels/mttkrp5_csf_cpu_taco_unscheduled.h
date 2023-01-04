#ifndef MTTKRP5_CPU_TACO_UNSCHEDULED_H
#define MTTKRP5_CPU_TACO_UNSCHEDULED_H

#include "ds.h"
#include "timers.h"

void mttkrp5_cpu_taco_unscheduled(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D, taco_tensor_t *E, taco_tensor_t *F) {
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
  splatt_idx_t* restrict B5_pos = (splatt_idx_t*)(B->indices[4][0]);
  splatt_idx_t* restrict B5_crd = (splatt_idx_t*)(B->indices[4][1]);
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
  splatt_idx_t F1_dimension = (splatt_idx_t)(F->dimensions[0]);
  splatt_idx_t F2_dimension = (splatt_idx_t)(F->dimensions[1]);
  double* restrict F_vals = (double*)(F->vals);

  _Pragma("omp parallel for schedule(static)")
  for (splatt_idx_t pA = 0; pA < (A1_dimension * A2_dimension); pA++) {
    A_vals[pA] = 0.0;
  }

  _Pragma("omp parallel for schedule(dynamic, 16)")
  for (splatt_idx_t i = 0; i < B1_dimension; i++) {
    for (splatt_idx_t pB2 = B2_pos[i]; pB2 < B2_pos[(i + 1)]; pB2++) {
      splatt_idx_t k = B2_crd[pB2];
      for (splatt_idx_t pB3 = B3_pos[pB2]; pB3 < B3_pos[(pB2 + 1)]; pB3++) {
        splatt_idx_t l = B3_crd[pB3];
        for (splatt_idx_t pB4 = B4_pos[pB3]; pB4 < B4_pos[(pB3 + 1)]; pB4++) {
          splatt_idx_t m = B4_crd[pB4];
          for (splatt_idx_t pB5 = B5_pos[pB4]; pB5 < B5_pos[(pB4 + 1)]; pB5++) {
            splatt_idx_t n = B5_crd[pB5];
            for (splatt_idx_t j = 0; j < F2_dimension; j++) {
              splatt_idx_t pA2 = i * A2_dimension + j;
              splatt_idx_t pC2 = k * C2_dimension + j;
              splatt_idx_t pD2 = l * D2_dimension + j;
              splatt_idx_t pE2 = m * E2_dimension + j;
              splatt_idx_t pF2 = n * F2_dimension + j;
              A_vals[pA2] = A_vals[pA2] + B_vals[pB5] * C_vals[pC2] * D_vals[pD2] * E_vals[pE2] * F_vals[pF2];
            }
          }
        }
      }
    }
  }
)
}

#endif
