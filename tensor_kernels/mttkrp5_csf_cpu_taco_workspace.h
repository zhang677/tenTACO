#ifndef MTTKRP5_CPU_TACO_WORKSPACE_H
#define MTTKRP5_CPU_TACO_WORKSPACE_H

#include "ds.h"
#include "timers.h"

void mttkrp5_cpu_taco_workspace(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D, taco_tensor_t *E, taco_tensor_t *F) {
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

  _Pragma("omp parallel for schedule(dynamic, 1)")
  for (splatt_idx_t i1 = 0; i1 < ((B1_dimension + 63) / 64); i1++) {
    double* restrict BDEF_workspace = 0;
    BDEF_workspace = (double*)malloc(sizeof(double) * C2_dimension);
    for (splatt_idx_t pBDEF_workspace = 0; pBDEF_workspace < C2_dimension; pBDEF_workspace++) {
      BDEF_workspace[pBDEF_workspace] = 0.0;
    }
    double* restrict BEF_workspace = 0;
    BEF_workspace = (double*)malloc(sizeof(double) * C2_dimension);
    for (splatt_idx_t pBEF_workspace = 0; pBEF_workspace < C2_dimension; pBEF_workspace++) {
      BEF_workspace[pBEF_workspace] = 0.0;
    }
    double* restrict BF_workspace = 0;
    BF_workspace = (double*)malloc(sizeof(double) * C2_dimension);
    for (splatt_idx_t pBF_workspace = 0; pBF_workspace < C2_dimension; pBF_workspace++) {
      BF_workspace[pBF_workspace] = 0.0;
    }
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
            for (splatt_idx_t nB = B5_pos[mB]; nB < B5_pos[(mB + 1)]; nB++) {
              splatt_idx_t n = B5_crd[nB];
              for (splatt_idx_t j = 0; j < C2_dimension; j++) {
                splatt_idx_t jF = n * F2_dimension + j;
                BF_workspace[j] = BF_workspace[j] + B_vals[nB] * F_vals[jF];
              }
            }
            for (splatt_idx_t j = 0; j < C2_dimension; j++) {
              splatt_idx_t jE = m * E2_dimension + j;
              BEF_workspace[j] = BEF_workspace[j] + BF_workspace[j] * E_vals[jE];
              BF_workspace[j] = 0;
            }
          }
          for (splatt_idx_t j = 0; j < C2_dimension; j++) {
            splatt_idx_t jD = l * D2_dimension + j;
            BDEF_workspace[j] = BDEF_workspace[j] + BEF_workspace[j] * D_vals[jD];
            BEF_workspace[j] = 0;
          }
        }
        for (splatt_idx_t j = 0; j < C2_dimension; j++) {
          splatt_idx_t jA = i * A2_dimension + j;
          splatt_idx_t jC = k * C2_dimension + j;
          A_vals[jA] = A_vals[jA] + BDEF_workspace[j] * C_vals[jC];
          BDEF_workspace[j] = 0;
        }
      }
    }
    free(BF_workspace);
    free(BEF_workspace);
    free(BDEF_workspace);
  }
);
}

#endif
