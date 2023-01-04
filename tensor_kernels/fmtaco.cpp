#include <vector>
#include <iostream>
#include <string>
#include <cstring>

#include "mex.h"
#include "matrix.h"

#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_dim_dense, taco_dim_sparse } taco_dim_t;
typedef struct {
  int32_t     order;      // tensor order (number of dimensions)
  int32_t*    dims;       // tensor dimensions
  taco_dim_t* dim_types;  // dimension storage types
  int32_t     csize;      // component size
  int32_t*    dim_order;  // dimension storage order
  uint8_t***  indices;    // tensor index data (per dimension)
  uint8_t*    vals;       // tensor values
} taco_tensor_t;
#define restrict  __restrict__
#endif
#endif

int compute_ttv(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *c) {
  int* restrict A0_pos_arr = (int*)(A->indices[0][0]);
  int* restrict A0_idx_arr = (int*)(A->indices[0][1]);
  int* restrict A1_idx_arr = (int*)(A->indices[1][0]);
  double* restrict A_val_arr = (double*)(A->vals);
  int* restrict B0_pos_arr = (int*)(B->indices[0][0]);
  int* restrict B0_idx_arr = (int*)(B->indices[0][1]);
  int* restrict B1_idx_arr = (int*)(B->indices[1][0]);
  int* restrict B2_idx_arr = (int*)(B->indices[2][0]);
  double* restrict B_val_arr = (double*)(B->vals);
  int c0_size = *(int*)(c->indices[0][0]);
  double* restrict c_val_arr = (double*)(c->vals);

  int32_t init_alloc_size = B0_pos_arr[1];
  int32_t A0_idx_capacity = init_alloc_size;
  A0_pos_arr = (int*)malloc(sizeof(int) * init_alloc_size);
  A0_idx_arr = (int*)malloc(sizeof(int) * A0_idx_capacity);
  A0_pos_arr[0] = 0;
  int32_t A1_idx_capacity = init_alloc_size;
  A1_idx_arr = (int*)malloc(sizeof(int) * A1_idx_capacity);
  int32_t A_vals_capacity = init_alloc_size;
  A_val_arr = (double*)malloc(sizeof(double) * A_vals_capacity);

  int32_t A0_pos = 0;
  int32_t A1_pos = 0;
  int32_t B0_pos = B0_pos_arr[0];
  while (B0_pos < B0_pos_arr[1]) {
    int32_t iB = B0_idx_arr[B0_pos];
    int32_t B0_end = B0_pos + 1;
    while ((B0_end < B0_pos_arr[1]) && (B0_idx_arr[B0_end] == iB)) {
      B0_end++;
    }
    int32_t A1_pos_start = A1_pos;
    if (A_vals_capacity <= (A0_pos + 1)) {
      int32_t A_vals_capacity_new = 2 * (A0_pos + 1);
      A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new);
      A_vals_capacity = A_vals_capacity_new;
    }
    int32_t B1_pos = B0_pos;
    while (B1_pos < B0_end) {
      int32_t jB = B1_idx_arr[B1_pos];
      int32_t B1_end = B1_pos + 1;
      while ((B1_end < B0_end) && (B1_idx_arr[B1_end] == jB)) {
        B1_end++;
      }
      double tk = 0;
      int32_t B2_pos = B1_pos;
      int32_t kc = 0;
      for (int32_t B2_pos = B1_pos; B2_pos < B1_end; B2_pos++) {
        int32_t kB = B2_idx_arr[B2_pos];
        int32_t B2_end = B2_pos + 1;
        int32_t c0_end = kB + 1;
        tk += B_val_arr[B2_pos] * c_val_arr[kB];
      }
      A_val_arr[A1_pos] = tk;
      if (A1_idx_capacity <= A1_pos) {
        A1_idx_capacity = 2 * A1_pos;
        A1_idx_arr = (int*)realloc(A1_idx_arr, sizeof(int) * A1_idx_capacity);
      }
      A1_idx_arr[A1_pos] = jB;
      A1_pos++;
      B1_pos = B1_end;
    }
    int32_t A1_pos_inserted = A1_pos - A1_pos_start;
    if (A1_pos_inserted > 0)
      for (int32_t it = 0; it < A1_pos_inserted; it++) {
        if (A0_idx_capacity <= A0_pos) {
          A0_idx_capacity = 2 * A0_pos;
          A0_idx_arr = (int*)realloc(A0_idx_arr, sizeof(int) * A0_idx_capacity);
        }
        A0_idx_arr[A0_pos] = iB;
        A0_pos++;
      }
    B0_pos = B0_end;
  }
  A0_pos_arr[1] = A0_pos;

  A->indices[0][0] = (uint8_t*)(A0_pos_arr);
  A->indices[0][1] = (uint8_t*)(A0_idx_arr);
  A->indices[1][0] = (uint8_t*)(A1_idx_arr);
  A->vals = (uint8_t*)A_val_arr;

  return 0;
}

int compute_ttm(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C) {
  int* restrict A0_pos_arr = (int*)(A->indices[0][0]);
  int* restrict A0_idx_arr = (int*)(A->indices[0][1]);
  int* restrict A1_idx_arr = (int*)(A->indices[1][0]);
  int* restrict A2_idx_arr = (int*)(A->indices[2][0]);
  double* restrict A_val_arr = (double*)(A->vals);
  int* restrict B0_pos_arr = (int*)(B->indices[0][0]);
  int* restrict B0_idx_arr = (int*)(B->indices[0][1]);
  int* restrict B1_idx_arr = (int*)(B->indices[1][0]);
  int* restrict B2_idx_arr = (int*)(B->indices[2][0]);
  double* restrict B_val_arr = (double*)(B->vals);
  int C0_size = *(int*)(C->indices[0][0]);
  int C1_size = *(int*)(C->indices[1][0]);
  double* restrict C_val_arr = (double*)(C->vals);

  int32_t init_alloc_size = 1048576;
  int32_t A0_pos_capacity = 2;
  int32_t A0_idx_capacity = init_alloc_size;
  A0_pos_arr = (int*)malloc(sizeof(int) * A0_pos_capacity);
  A0_idx_arr = (int*)malloc(sizeof(int) * A0_idx_capacity);
  A0_pos_arr[0] = 0;
  int32_t A1_idx_capacity = init_alloc_size;
  A1_idx_arr = (int*)malloc(sizeof(int) * A1_idx_capacity);
  int32_t A2_idx_capacity = init_alloc_size;
  A2_idx_arr = (int*)malloc(sizeof(int) * A2_idx_capacity);
  int32_t A_vals_capacity = init_alloc_size;
  A_val_arr = (double*)malloc(sizeof(double) * A_vals_capacity);

  int32_t A0_pos = 0;
  int32_t A1_pos = 0;
  int32_t A2_pos = 0;
  int32_t B0_pos = B0_pos_arr[0];
  while (B0_pos < B0_pos_arr[1]) {
    int32_t iB = B0_idx_arr[B0_pos];
    int32_t B0_end = B0_pos + 1;
    while ((B0_end < B0_pos_arr[1]) && (B0_idx_arr[B0_end] == iB)) {
      B0_end++;
    }
    int32_t A1_pos_start = A1_pos;
    int32_t B1_pos = B0_pos;
    while (B1_pos < B0_end) {
      int32_t jB = B1_idx_arr[B1_pos];
      int32_t B1_end = B1_pos + 1;
      while ((B1_end < B0_end) && (B1_idx_arr[B1_end] == jB)) {
        B1_end++;
      }
      int32_t A2_pos_start = A2_pos;
      int32_t kC = 0;
      for (int32_t kC = 0; kC < C0_size; kC++) {
        int32_t C0_pos = (0 * C0_size) + kC;
        int32_t C0_end = C0_pos + 1;
        if (A_vals_capacity <= ((A2_pos + 1) * 1)) {
          int32_t A_vals_capacity_new = 2 * ((A2_pos + 1) * 1);
          A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new);
          A_vals_capacity = A_vals_capacity_new;
        }
        double tl = 0;
        int32_t B2_pos = B1_pos;
        int32_t lC = 0;
        for (int32_t B2_pos = B1_pos; B2_pos < B1_end; B2_pos++) {
          int32_t lB = B2_idx_arr[B2_pos];
          int32_t C1_pos = (C0_pos * C1_size) + lB;
          int32_t B2_end = B2_pos + 1;
          int32_t C1_end = C1_pos + 1;
          tl += B_val_arr[B2_pos] * C_val_arr[C1_pos];
        }
        A_val_arr[A2_pos] = tl;
        if (A2_idx_capacity <= A2_pos) {
          A2_idx_capacity = 2 * A2_pos;
          A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
        }
        A2_idx_arr[A2_pos] = kC;
        A2_pos++;
      }
      int32_t A2_pos_inserted = A2_pos - A2_pos_start;
      if (A2_pos_inserted > 0)
        for (int32_t it = 0; it < A2_pos_inserted; it++) {
          if (A1_idx_capacity <= A1_pos) {
            A1_idx_capacity = 2 * A1_pos;
            A1_idx_arr = (int*)realloc(A1_idx_arr, sizeof(int) * A1_idx_capacity);
          }
          A1_idx_arr[A1_pos] = jB;
          A1_pos++;
        }
      B1_pos = B1_end;
    }
    int32_t A1_pos_inserted = A1_pos - A1_pos_start;
    if (A1_pos_inserted > 0)
      for (int32_t it0 = 0; it0 < A1_pos_inserted; it0++) {
        if (A0_idx_capacity <= A0_pos) {
          A0_idx_capacity = 2 * A0_pos;
          A0_idx_arr = (int*)realloc(A0_idx_arr, sizeof(int) * A0_idx_capacity);
        }
        A0_idx_arr[A0_pos] = iB;
        A0_pos++;
      }
    B0_pos = B0_end;
  }
  A0_pos_arr[(0 + 1)] = A0_pos;

  A->indices[0][0] = (uint8_t*)(A0_pos_arr);
  A->indices[0][1] = (uint8_t*)(A0_idx_arr);
  A->indices[1][0] = (uint8_t*)(A1_idx_arr);
  A->indices[2][0] = (uint8_t*)(A2_idx_arr);
  A->vals = (uint8_t*)A_val_arr;
  return 0;
}

int compute_plus(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C) {
  int* restrict A0_pos_arr = (int*)(A->indices[0][0]);
  int* restrict A0_idx_arr = (int*)(A->indices[0][1]);
  int* restrict A1_idx_arr = (int*)(A->indices[1][0]);
  int* restrict A2_idx_arr = (int*)(A->indices[2][0]);
  double* restrict A_val_arr = (double*)(A->vals);
  int* restrict B0_pos_arr = (int*)(B->indices[0][0]);
  int* restrict B0_idx_arr = (int*)(B->indices[0][1]);
  int* restrict B1_idx_arr = (int*)(B->indices[1][0]);
  int* restrict B2_idx_arr = (int*)(B->indices[2][0]);
  double* restrict B_val_arr = (double*)(B->vals);
  int* restrict C0_pos_arr = (int*)(C->indices[0][0]);
  int* restrict C0_idx_arr = (int*)(C->indices[0][1]);
  int* restrict C1_idx_arr = (int*)(C->indices[1][0]);
  int* restrict C2_idx_arr = (int*)(C->indices[2][0]);
  double* restrict C_val_arr = (double*)(C->vals);

  //int32_t init_alloc_size = 1048576;
  int32_t init_alloc_size = B0_pos_arr[1] + C0_pos_arr[1];
  int32_t A0_idx_capacity = init_alloc_size;
  A0_pos_arr = (int*)malloc(sizeof(int) * init_alloc_size);
  A0_idx_arr = (int*)malloc(sizeof(int) * A0_idx_capacity);
  A0_pos_arr[0] = 0;
  int32_t A1_idx_capacity = init_alloc_size;
  A1_idx_arr = (int*)malloc(sizeof(int) * A1_idx_capacity);
  int32_t A2_idx_capacity = init_alloc_size;
  A2_idx_arr = (int*)malloc(sizeof(int) * A2_idx_capacity);
  int32_t A_vals_capacity = init_alloc_size;
  A_val_arr = (double*)malloc(sizeof(double) * A_vals_capacity);

  int32_t A0_pos = 0;
  int32_t A1_pos = 0;
  int32_t A2_pos = 0;
  int32_t B0_pos = B0_pos_arr[0];
  int32_t C0_pos = C0_pos_arr[0];
  while ((B0_pos < B0_pos_arr[1]) && (C0_pos < C0_pos_arr[1])) {
    int32_t iB = B0_idx_arr[B0_pos];
    int32_t iC = C0_idx_arr[C0_pos];
    int32_t i = TACO_MIN(iB,iC);
    int32_t B0_end = B0_pos + 1;
    if (iB == i)
      while ((B0_end < B0_pos_arr[1]) && (B0_idx_arr[B0_end] == i)) {
        B0_end++;
      }
    int32_t C0_end = C0_pos + 1;
    if (iC == i)
      while ((C0_end < C0_pos_arr[1]) && (C0_idx_arr[C0_end] == i)) {
        C0_end++;
      }
    int32_t A1_pos_start = A1_pos;
    if ((iB == i) && (iC == i)) {
      int32_t B1_pos = B0_pos;
      int32_t C1_pos = C0_pos;
      while ((B1_pos < B0_end) && (C1_pos < C0_end)) {
        int32_t jB = B1_idx_arr[B1_pos];
        int32_t jC = C1_idx_arr[C1_pos];
        int32_t j = TACO_MIN(jB,jC);
        int32_t B1_end = B1_pos + 1;
        if (jB == j)
          while ((B1_end < B0_end) && (B1_idx_arr[B1_end] == j)) {
            B1_end++;
          }
        int32_t C1_end = C1_pos + 1;
        if (jC == j)
          while ((C1_end < C0_end) && (C1_idx_arr[C1_end] == j)) {
            C1_end++;
          }
        int32_t A2_pos_start = A2_pos;
        if ((jB == j) && (jC == j)) {
          int32_t B2_pos = B1_pos;
          int32_t C2_pos = C1_pos;
          while ((B2_pos < B1_end) && (C2_pos < C1_end)) {
            int32_t kB = B2_idx_arr[B2_pos];
            int32_t kC = C2_idx_arr[C2_pos];
            int32_t k = TACO_MIN(kB,kC);
            int32_t B2_end = B2_pos + 1;
            int32_t C2_end = C2_pos + 1;
            if (A_vals_capacity <= (A2_pos + 1)) {
              int32_t A_vals_capacity_new = 2 * (A2_pos + 1);
              A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new);
              A_vals_capacity = A_vals_capacity_new;
            }
            if ((kB == k) && (kC == k)) {
              A_val_arr[A2_pos] = B_val_arr[B2_pos] + C_val_arr[C2_pos];
              if (A2_idx_capacity <= A2_pos) {
                A2_idx_capacity = 2 * A2_pos;
                A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
              }
              A2_idx_arr[A2_pos] = k;
              A2_pos++;
            }
            else if (kB == k) {
              A_val_arr[A2_pos] = B_val_arr[B2_pos];
              if (A2_idx_capacity <= A2_pos) {
                A2_idx_capacity = 2 * A2_pos;
                A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
              }
              A2_idx_arr[A2_pos] = k;
              A2_pos++;
            }
            else {
              A_val_arr[A2_pos] = C_val_arr[C2_pos];
              if (A2_idx_capacity <= A2_pos) {
                A2_idx_capacity = 2 * A2_pos;
                A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
              }
              A2_idx_arr[A2_pos] = k;
              A2_pos++;
            }
            if (kB == k) B2_pos = B2_end;
            if (kC == k) C2_pos = C2_end;
          }
          while (B2_pos < B1_end) {
            int32_t kB = B2_idx_arr[B2_pos];
            int32_t B2_end = B2_pos + 1;
            if (A_vals_capacity <= (A2_pos + 1)) {
              int32_t A_vals_capacity_new0 = 2 * (A2_pos + 1);
              A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new0);
              A_vals_capacity = A_vals_capacity_new0;
            }
            A_val_arr[A2_pos] = B_val_arr[B2_pos];
            if (A2_idx_capacity <= A2_pos) {
              A2_idx_capacity = 2 * A2_pos;
              A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
            }
            A2_idx_arr[A2_pos] = kB;
            A2_pos++;
            B2_pos = B2_end;
          }
          while (C2_pos < C1_end) {
            int32_t kC = C2_idx_arr[C2_pos];
            int32_t C2_end = C2_pos + 1;
            if (A_vals_capacity <= (A2_pos + 1)) {
              int32_t A_vals_capacity_new1 = 2 * (A2_pos + 1);
              A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new1);
              A_vals_capacity = A_vals_capacity_new1;
            }
            A_val_arr[A2_pos] = C_val_arr[C2_pos];
            if (A2_idx_capacity <= A2_pos) {
              A2_idx_capacity = 2 * A2_pos;
              A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
            }
            A2_idx_arr[A2_pos] = kC;
            A2_pos++;
            C2_pos = C2_end;
          }
          int32_t A2_pos_inserted = A2_pos - A2_pos_start;
          if (A2_pos_inserted > 0)
            for (int32_t it = 0; it < A2_pos_inserted; it++) {
              if (A1_idx_capacity <= A1_pos) {
                A1_idx_capacity = 2 * A1_pos;
                A1_idx_arr = (int*)realloc(A1_idx_arr, sizeof(int) * A1_idx_capacity);
              }
              A1_idx_arr[A1_pos] = j;
              A1_pos++;
            }
        }
        else if (jB == j) {
          for (int32_t B2_pos = B1_pos; B2_pos < B1_end; B2_pos++) {
            int32_t kB = B2_idx_arr[B2_pos];
            int32_t B2_end = B2_pos + 1;
            if (A_vals_capacity <= (A2_pos + 1)) {
              int32_t A_vals_capacity_new2 = 2 * (A2_pos + 1);
              A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new2);
              A_vals_capacity = A_vals_capacity_new2;
            }
            A_val_arr[A2_pos] = B_val_arr[B2_pos];
            if (A2_idx_capacity <= A2_pos) {
              A2_idx_capacity = 2 * A2_pos;
              A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
            }
            A2_idx_arr[A2_pos] = kB;
            A2_pos++;
          }
          int32_t A2_pos_inserted0 = A2_pos - A2_pos_start;
          if (A2_pos_inserted0 > 0)
            for (int32_t it0 = 0; it0 < A2_pos_inserted0; it0++) {
              if (A1_idx_capacity <= A1_pos) {
                A1_idx_capacity = 2 * A1_pos;
                A1_idx_arr = (int*)realloc(A1_idx_arr, sizeof(int) * A1_idx_capacity);
              }
              A1_idx_arr[A1_pos] = j;
              A1_pos++;
            }
        }
        else {
          for (int32_t C2_pos = C1_pos; C2_pos < C1_end; C2_pos++) {
            int32_t kC = C2_idx_arr[C2_pos];
            int32_t C2_end = C2_pos + 1;
            if (A_vals_capacity <= (A2_pos + 1)) {
              int32_t A_vals_capacity_new3 = 2 * (A2_pos + 1);
              A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new3);
              A_vals_capacity = A_vals_capacity_new3;
            }
            A_val_arr[A2_pos] = C_val_arr[C2_pos];
            if (A2_idx_capacity <= A2_pos) {
              A2_idx_capacity = 2 * A2_pos;
              A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
            }
            A2_idx_arr[A2_pos] = kC;
            A2_pos++;
          }
          int32_t A2_pos_inserted1 = A2_pos - A2_pos_start;
          if (A2_pos_inserted1 > 0)
            for (int32_t it1 = 0; it1 < A2_pos_inserted1; it1++) {
              if (A1_idx_capacity <= A1_pos) {
                A1_idx_capacity = 2 * A1_pos;
                A1_idx_arr = (int*)realloc(A1_idx_arr, sizeof(int) * A1_idx_capacity);
              }
              A1_idx_arr[A1_pos] = j;
              A1_pos++;
            }
        }
        if (jB == j) B1_pos = B1_end;
        if (jC == j) C1_pos = C1_end;
      }
      while (B1_pos < B0_end) {
        int32_t jB = B1_idx_arr[B1_pos];
        int32_t B1_end = B1_pos + 1;
        while ((B1_end < B0_end) && (B1_idx_arr[B1_end] == jB)) {
          B1_end++;
        }
        int32_t A2_pos_start0 = A2_pos;
        for (int32_t B2_pos = B1_pos; B2_pos < B1_end; B2_pos++) {
          int32_t kB = B2_idx_arr[B2_pos];
          int32_t B2_end = B2_pos + 1;
          if (A_vals_capacity <= (A2_pos + 1)) {
            int32_t A_vals_capacity_new4 = 2 * (A2_pos + 1);
            A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new4);
            A_vals_capacity = A_vals_capacity_new4;
          }
          A_val_arr[A2_pos] = B_val_arr[B2_pos];
          if (A2_idx_capacity <= A2_pos) {
            A2_idx_capacity = 2 * A2_pos;
            A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
          }
          A2_idx_arr[A2_pos] = kB;
          A2_pos++;
        }
        int32_t A2_pos_inserted2 = A2_pos - A2_pos_start0;
        if (A2_pos_inserted2 > 0)
          for (int32_t it2 = 0; it2 < A2_pos_inserted2; it2++) {
            if (A1_idx_capacity <= A1_pos) {
              A1_idx_capacity = 2 * A1_pos;
              A1_idx_arr = (int*)realloc(A1_idx_arr, sizeof(int) * A1_idx_capacity);
            }
            A1_idx_arr[A1_pos] = jB;
            A1_pos++;
          }
        B1_pos = B1_end;
      }
      while (C1_pos < C0_end) {
        int32_t jC = C1_idx_arr[C1_pos];
        int32_t C1_end = C1_pos + 1;
        while ((C1_end < C0_end) && (C1_idx_arr[C1_end] == jC)) {
          C1_end++;
        }
        int32_t A2_pos_start1 = A2_pos;
        for (int32_t C2_pos = C1_pos; C2_pos < C1_end; C2_pos++) {
          int32_t kC = C2_idx_arr[C2_pos];
          int32_t C2_end = C2_pos + 1;
          if (A_vals_capacity <= (A2_pos + 1)) {
            int32_t A_vals_capacity_new5 = 2 * (A2_pos + 1);
            A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new5);
            A_vals_capacity = A_vals_capacity_new5;
          }
          A_val_arr[A2_pos] = C_val_arr[C2_pos];
          if (A2_idx_capacity <= A2_pos) {
            A2_idx_capacity = 2 * A2_pos;
            A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
          }
          A2_idx_arr[A2_pos] = kC;
          A2_pos++;
        }
        int32_t A2_pos_inserted3 = A2_pos - A2_pos_start1;
        if (A2_pos_inserted3 > 0)
          for (int32_t it3 = 0; it3 < A2_pos_inserted3; it3++) {
            if (A1_idx_capacity <= A1_pos) {
              A1_idx_capacity = 2 * A1_pos;
              A1_idx_arr = (int*)realloc(A1_idx_arr, sizeof(int) * A1_idx_capacity);
            }
            A1_idx_arr[A1_pos] = jC;
            A1_pos++;
          }
        C1_pos = C1_end;
      }
      int32_t A1_pos_inserted = A1_pos - A1_pos_start;
      if (A1_pos_inserted > 0)
        for (int32_t it4 = 0; it4 < A1_pos_inserted; it4++) {
          if (A0_idx_capacity <= A0_pos) {
            A0_idx_capacity = 2 * A0_pos;
            A0_idx_arr = (int*)realloc(A0_idx_arr, sizeof(int) * A0_idx_capacity);
          }
          A0_idx_arr[A0_pos] = i;
          A0_pos++;
        }
    }
    else if (iB == i) {
      int32_t B1_pos = B0_pos;
      while (B1_pos < B0_end) {
        int32_t jB = B1_idx_arr[B1_pos];
        int32_t B1_end = B1_pos + 1;
        while ((B1_end < B0_end) && (B1_idx_arr[B1_end] == jB)) {
          B1_end++;
        }
        int32_t A2_pos_start2 = A2_pos;
        for (int32_t B2_pos = B1_pos; B2_pos < B1_end; B2_pos++) {
          int32_t kB = B2_idx_arr[B2_pos];
          int32_t B2_end = B2_pos + 1;
          if (A_vals_capacity <= (A2_pos + 1)) {
            int32_t A_vals_capacity_new6 = 2 * (A2_pos + 1);
            A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new6);
            A_vals_capacity = A_vals_capacity_new6;
          }
          A_val_arr[A2_pos] = B_val_arr[B2_pos];
          if (A2_idx_capacity <= A2_pos) {
            A2_idx_capacity = 2 * A2_pos;
            A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
          }
          A2_idx_arr[A2_pos] = kB;
          A2_pos++;
        }
        int32_t A2_pos_inserted4 = A2_pos - A2_pos_start2;
        if (A2_pos_inserted4 > 0)
          for (int32_t it5 = 0; it5 < A2_pos_inserted4; it5++) {
            if (A1_idx_capacity <= A1_pos) {
              A1_idx_capacity = 2 * A1_pos;
              A1_idx_arr = (int*)realloc(A1_idx_arr, sizeof(int) * A1_idx_capacity);
            }
            A1_idx_arr[A1_pos] = jB;
            A1_pos++;
          }
        B1_pos = B1_end;
      }
      int32_t A1_pos_inserted0 = A1_pos - A1_pos_start;
      if (A1_pos_inserted0 > 0)
        for (int32_t it6 = 0; it6 < A1_pos_inserted0; it6++) {
          if (A0_idx_capacity <= A0_pos) {
            A0_idx_capacity = 2 * A0_pos;
            A0_idx_arr = (int*)realloc(A0_idx_arr, sizeof(int) * A0_idx_capacity);
          }
          A0_idx_arr[A0_pos] = i;
          A0_pos++;
        }
    }
    else {
      int32_t C1_pos = C0_pos;
      while (C1_pos < C0_end) {
        int32_t jC = C1_idx_arr[C1_pos];
        int32_t C1_end = C1_pos + 1;
        while ((C1_end < C0_end) && (C1_idx_arr[C1_end] == jC)) {
          C1_end++;
        }
        int32_t A2_pos_start3 = A2_pos;
        for (int32_t C2_pos = C1_pos; C2_pos < C1_end; C2_pos++) {
          int32_t kC = C2_idx_arr[C2_pos];
          int32_t C2_end = C2_pos + 1;
          if (A_vals_capacity <= (A2_pos + 1)) {
            int32_t A_vals_capacity_new7 = 2 * (A2_pos + 1);
            A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new7);
            A_vals_capacity = A_vals_capacity_new7;
          }
          A_val_arr[A2_pos] = C_val_arr[C2_pos];
          if (A2_idx_capacity <= A2_pos) {
            A2_idx_capacity = 2 * A2_pos;
            A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
          }
          A2_idx_arr[A2_pos] = kC;
          A2_pos++;
        }
        int32_t A2_pos_inserted5 = A2_pos - A2_pos_start3;
        if (A2_pos_inserted5 > 0)
          for (int32_t it7 = 0; it7 < A2_pos_inserted5; it7++) {
            if (A1_idx_capacity <= A1_pos) {
              A1_idx_capacity = 2 * A1_pos;
              A1_idx_arr = (int*)realloc(A1_idx_arr, sizeof(int) * A1_idx_capacity);
            }
            A1_idx_arr[A1_pos] = jC;
            A1_pos++;
          }
        C1_pos = C1_end;
      }
      int32_t A1_pos_inserted1 = A1_pos - A1_pos_start;
      if (A1_pos_inserted1 > 0)
        for (int32_t it8 = 0; it8 < A1_pos_inserted1; it8++) {
          if (A0_idx_capacity <= A0_pos) {
            A0_idx_capacity = 2 * A0_pos;
            A0_idx_arr = (int*)realloc(A0_idx_arr, sizeof(int) * A0_idx_capacity);
          }
          A0_idx_arr[A0_pos] = i;
          A0_pos++;
        }
    }
    if (iB == i) B0_pos = B0_end;
    if (iC == i) C0_pos = C0_end;
  }
  while (B0_pos < B0_pos_arr[1]) {
    int32_t iB = B0_idx_arr[B0_pos];
    int32_t B0_end = B0_pos + 1;
    while ((B0_end < B0_pos_arr[1]) && (B0_idx_arr[B0_end] == iB)) {
      B0_end++;
    }
    int32_t A1_pos_start0 = A1_pos;
    int32_t B1_pos = B0_pos;
    while (B1_pos < B0_end) {
      int32_t jB = B1_idx_arr[B1_pos];
      int32_t B1_end = B1_pos + 1;
      while ((B1_end < B0_end) && (B1_idx_arr[B1_end] == jB)) {
        B1_end++;
      }
      int32_t A2_pos_start4 = A2_pos;
      for (int32_t B2_pos = B1_pos; B2_pos < B1_end; B2_pos++) {
        int32_t kB = B2_idx_arr[B2_pos];
        int32_t B2_end = B2_pos + 1;
        if (A_vals_capacity <= (A2_pos + 1)) {
          int32_t A_vals_capacity_new8 = 2 * (A2_pos + 1);
          A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new8);
          A_vals_capacity = A_vals_capacity_new8;
        }
        A_val_arr[A2_pos] = B_val_arr[B2_pos];
        if (A2_idx_capacity <= A2_pos) {
          A2_idx_capacity = 2 * A2_pos;
          A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
        }
        A2_idx_arr[A2_pos] = kB;
        A2_pos++;
      }
      int32_t A2_pos_inserted6 = A2_pos - A2_pos_start4;
      if (A2_pos_inserted6 > 0)
        for (int32_t it9 = 0; it9 < A2_pos_inserted6; it9++) {
          if (A1_idx_capacity <= A1_pos) {
            A1_idx_capacity = 2 * A1_pos;
            A1_idx_arr = (int*)realloc(A1_idx_arr, sizeof(int) * A1_idx_capacity);
          }
          A1_idx_arr[A1_pos] = jB;
          A1_pos++;
        }
      B1_pos = B1_end;
    }
    int32_t A1_pos_inserted2 = A1_pos - A1_pos_start0;
    if (A1_pos_inserted2 > 0)
      for (int32_t it10 = 0; it10 < A1_pos_inserted2; it10++) {
        if (A0_idx_capacity <= A0_pos) {
          A0_idx_capacity = 2 * A0_pos;
          A0_idx_arr = (int*)realloc(A0_idx_arr, sizeof(int) * A0_idx_capacity);
        }
        A0_idx_arr[A0_pos] = iB;
        A0_pos++;
      }
    B0_pos = B0_end;
  }
  while (C0_pos < C0_pos_arr[1]) {
    int32_t iC = C0_idx_arr[C0_pos];
    int32_t C0_end = C0_pos + 1;
    while ((C0_end < C0_pos_arr[1]) && (C0_idx_arr[C0_end] == iC)) {
      C0_end++;
    }
    int32_t A1_pos_start1 = A1_pos;
    int32_t C1_pos = C0_pos;
    while (C1_pos < C0_end) {
      int32_t jC = C1_idx_arr[C1_pos];
      int32_t C1_end = C1_pos + 1;
      while ((C1_end < C0_end) && (C1_idx_arr[C1_end] == jC)) {
        C1_end++;
      }
      int32_t A2_pos_start5 = A2_pos;
      for (int32_t C2_pos = C1_pos; C2_pos < C1_end; C2_pos++) {
        int32_t kC = C2_idx_arr[C2_pos];
        int32_t C2_end = C2_pos + 1;
        if (A_vals_capacity <= (A2_pos + 1)) {
          int32_t A_vals_capacity_new9 = 2 * (A2_pos + 1);
          A_val_arr = (double*)realloc(A_val_arr, sizeof(double) * A_vals_capacity_new9);
          A_vals_capacity = A_vals_capacity_new9;
        }
        A_val_arr[A2_pos] = C_val_arr[C2_pos];
        if (A2_idx_capacity <= A2_pos) {
          A2_idx_capacity = 2 * A2_pos;
          A2_idx_arr = (int*)realloc(A2_idx_arr, sizeof(int) * A2_idx_capacity);
        }
        A2_idx_arr[A2_pos] = kC;
        A2_pos++;
      }
      int32_t A2_pos_inserted7 = A2_pos - A2_pos_start5;
      if (A2_pos_inserted7 > 0)
        for (int32_t it11 = 0; it11 < A2_pos_inserted7; it11++) {
          if (A1_idx_capacity <= A1_pos) {
            A1_idx_capacity = 2 * A1_pos;
            A1_idx_arr = (int*)realloc(A1_idx_arr, sizeof(int) * A1_idx_capacity);
          }
          A1_idx_arr[A1_pos] = jC;
          A1_pos++;
        }
      C1_pos = C1_end;
    }
    int32_t A1_pos_inserted3 = A1_pos - A1_pos_start1;
    if (A1_pos_inserted3 > 0)
      for (int32_t it12 = 0; it12 < A1_pos_inserted3; it12++) {
        if (A0_idx_capacity <= A0_pos) {
          A0_idx_capacity = 2 * A0_pos;
          A0_idx_arr = (int*)realloc(A0_idx_arr, sizeof(int) * A0_idx_capacity);
        }
        A0_idx_arr[A0_pos] = iC;
        A0_pos++;
      }
    C0_pos = C0_end;
  }
  A0_pos_arr[1] = A0_pos;

  A->indices[0][0] = (uint8_t*)(A0_pos_arr);
  A->indices[0][1] = (uint8_t*)(A0_idx_arr);
  A->indices[1][0] = (uint8_t*)(A1_idx_arr);
  A->indices[2][0] = (uint8_t*)(A2_idx_arr);
  A->vals = (uint8_t*)A_val_arr;
  return 0;
}

int compute_mttkrp(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
  int A0_size = *(int*)(A->indices[0][0]);
  int A1_size = *(int*)(A->indices[1][0]);
  double* restrict A_val_arr = (double*)(A->vals);
  int* restrict B0_pos_arr = (int*)(B->indices[0][0]);
  int* restrict B0_idx_arr = (int*)(B->indices[0][1]);
  int* restrict B1_idx_arr = (int*)(B->indices[1][0]);
  int* restrict B2_idx_arr = (int*)(B->indices[2][0]);
  double* restrict B_val_arr = (double*)(B->vals);
  int C0_size = *(int*)(C->indices[0][0]);
  int C1_size = *(int*)(C->indices[1][0]);
  double* restrict C_val_arr = (double*)(C->vals);
  int D0_size = *(int*)(D->indices[0][0]);
  int D1_size = *(int*)(D->indices[1][0]);
  double* restrict D_val_arr = (double*)(D->vals);

  int32_t A_vals_capacity = A0_size * A1_size;
  A_val_arr = (double*)malloc(sizeof(double) * A_vals_capacity);

  for (int32_t A_pos = 0; A_pos < (A0_size * A1_size); A_pos++) {
    A_val_arr[A_pos] = 0;
  }
  for (int32_t B0_pos = B0_pos_arr[0]; B0_pos < B0_pos_arr[1]; ++B0_pos) {
    int32_t iB = B0_idx_arr[B0_pos];
    int32_t kB = B1_idx_arr[B0_pos];
    int32_t lB = B2_idx_arr[B0_pos];
    double tl = B_val_arr[B0_pos];
    for (int32_t jC = 0; jC < C1_size; jC++) {
      int32_t C1_pos = (kB * C1_size) + jC;
      int32_t D1_pos = (lB * D1_size) + jC;
      int32_t A1_pos = (iB * A1_size) + jC;
      int32_t C1_end = C1_pos + 1;
      int32_t D1_end = D1_pos + 1;
      A_val_arr[A1_pos] = A_val_arr[A1_pos] + ((tl * C_val_arr[C1_pos]) * D_val_arr[D1_pos]);
    }
  }

  A->vals = (uint8_t*)A_val_arr;
  return 0;
}

int compute_innerprod(taco_tensor_t *a, taco_tensor_t *B, taco_tensor_t *C) {
  double* restrict a_val_arr = (double*)(a->vals);
  int* restrict B0_pos_arr = (int*)(B->indices[0][0]);
  int* restrict B0_idx_arr = (int*)(B->indices[0][1]);
  int* restrict B1_idx_arr = (int*)(B->indices[1][0]);
  int* restrict B2_idx_arr = (int*)(B->indices[2][0]);
  double* restrict B_val_arr = (double*)(B->vals);
  int* restrict C0_pos_arr = (int*)(C->indices[0][0]);
  int* restrict C0_idx_arr = (int*)(C->indices[0][1]);
  int* restrict C1_idx_arr = (int*)(C->indices[1][0]);
  int* restrict C2_idx_arr = (int*)(C->indices[2][0]);
  double* restrict C_val_arr = (double*)(C->vals);

  int32_t a_vals_capacity = 1;
  a_val_arr = (double*)malloc(sizeof(double) * a_vals_capacity);

  a_val_arr[0] = 0;
  int32_t B0_pos = B0_pos_arr[0];
  int32_t C0_pos = C0_pos_arr[0];
  while ((B0_pos < B0_pos_arr[1]) && (C0_pos < C0_pos_arr[1])) {
    int32_t iB = B0_idx_arr[B0_pos * 1];
    int32_t iC = C0_idx_arr[C0_pos * 1];
    int32_t i = TACO_MIN(iB,iC);
    int32_t B0_end = B0_pos + 1;
    if (iB == i)
      while ((B0_end < B0_pos_arr[1]) && (B0_idx_arr[B0_end * 1] == i)) {
        B0_end++;
      }
    int32_t C0_end = C0_pos + 1;
    if (iC == i)
      while ((C0_end < C0_pos_arr[1]) && (C0_idx_arr[C0_end * 1] == i)) {
        C0_end++;
      }
    if ((iB == i) && (iC == i)) {
      double tj = 0;
      int32_t B1_pos = B0_pos;
      int32_t C1_pos = C0_pos;
      while ((B1_pos < B0_end) && (C1_pos < C0_end)) {
        int32_t jB = B1_idx_arr[(B1_pos * 1) + 0];
        int32_t jC = C1_idx_arr[(C1_pos * 1) + 0];
        int32_t j = TACO_MIN(jB,jC);
        int32_t B1_end = B1_pos + 1;
        if (jB == j)
          while ((B1_end < B0_end) && (B1_idx_arr[(B1_end * 1) + 0] == j)) {
            B1_end++;
          }
        int32_t C1_end = C1_pos + 1;
        if (jC == j)
          while ((C1_end < C0_end) && (C1_idx_arr[(C1_end * 1) + 0] == j)) {
            C1_end++;
          }
        if ((jB == j) && (jC == j)) {
          double tk = 0;
          int32_t B2_pos = B1_pos;
          int32_t C2_pos = C1_pos;
          while ((B2_pos < B1_end) && (C2_pos < C1_end)) {
            int32_t kB = B2_idx_arr[(B2_pos * 1) + 0];
            int32_t kC = C2_idx_arr[(C2_pos * 1) + 0];
            int32_t k = TACO_MIN(kB,kC);
            int32_t B2_end = B2_pos + 1;
            int32_t C2_end = C2_pos + 1;
            if ((kB == k) && (kC == k)) {
              tk += B_val_arr[B2_pos] * C_val_arr[C2_pos];
            }
            if (kB == k) B2_pos = B2_end;
            if (kC == k) C2_pos = C2_end;
          }
          tj += tk;
        }
        if (jB == j) B1_pos = B1_end;
        if (jC == j) C1_pos = C1_end;
      }
      a_val_arr[0] = a_val_arr[0] + tj;
    }
    if (iB == i) B0_pos = B0_end;
    if (iC == i) C0_pos = C0_end;
  }

  a->vals = (uint8_t*)a_val_arr;
  return 0;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    /* check for proper number of arguments */
    if (nrhs % 3 != 1) {
        mexErrMsgIdAndTxt("taco:nrhs", "Number of inputs must be a multiple of three plus one.");
    } else if (nlhs > 2) {
        mexErrMsgIdAndTxt("taco:nlhs", "Cannot have more than two outputs.");
    }

    taco_tensor_t              output;
    std::vector<taco_tensor_t> inputs;

    for (size_t i = 0; i < nrhs - 1; i += 3) {
      if (!mxIsClass(prhs[i], "int32") && mxGetNumberOfElements(prhs[i]) > 0) {
        mexErrMsgIdAndTxt("taco:notInt", "Indices must be 32-bit integers.");
      } else if (!mxIsDouble(prhs[i + 1]) || mxIsComplex(prhs[i + 1])) {
        mexErrMsgIdAndTxt("taco:notDouble", "Values must be doubles.");
      } else if (mxGetN(prhs[i + 1]) != 1 && mxGetNumberOfElements(prhs[i]) > 0) {
        mexErrMsgIdAndTxt("taco:notColVector","Values must be column vector.");
      } else if (!mxIsClass(prhs[i + 2], "int32")) {
        mexErrMsgIdAndTxt("taco:notInt", "Dimensions must be 32-bit integers.");
      } else if (mxGetM(prhs[i + 2]) != 1) {
        mexErrMsgIdAndTxt("taco:notRowVector","Dimensions must be row vector.");
      } else if (mxGetM(prhs[i]) != mxGetM(prhs[i + 1]) && mxGetN(prhs[i]) != mxGetM(prhs[i + 1]) && mxGetM(prhs[i]) != 0) {
        mexErrMsgIdAndTxt("taco:invalidCoords", "Must have same number of coordinates and values.");
      } else if (mxGetN(prhs[i]) != mxGetN(prhs[i + 2]) && mxGetM(prhs[i]) != mxGetN(prhs[i + 2]) && mxGetN(prhs[i]) != 0) {
        mexErrMsgIdAndTxt("taco:invalidDims", "Invalid dimensions.");
      }

      const size_t order = mxGetN(prhs[i + 2]);
      uint8_t*** indices = (uint8_t***)mxMalloc(sizeof(uint8_t**) * order);

      inputs.emplace_back();
      inputs.back().order = (int32_t)order;
      inputs.back().dims = (int32_t*)mxGetData(prhs[i + 2]);
      inputs.back().vals = (uint8_t*)mxGetPr(prhs[i + 1]);
      inputs.back().indices = indices;

      if (mxGetN(prhs[i]) > 0) {
        const size_t nnz = mxGetM(prhs[i + 1]);

        indices[0] = (uint8_t**)mxMalloc(sizeof(uint8_t*) * 2);
        indices[0][0] = (uint8_t*)mxMalloc(sizeof(int32_t) * 2);
        ((int32_t*)(indices[0][0]))[0] = 0;
        ((int32_t*)(indices[0][0]))[1] = (int32_t)nnz;
        indices[0][1] = (uint8_t*)mxGetData(prhs[i]);
        
        for (size_t j = 1; j < order; ++j) {
          indices[j] = (uint8_t**)mxMalloc(sizeof(uint8_t*));
          indices[j][0] = (uint8_t*)(((int32_t*)indices[0][1]) + j * nnz);
        }
      } else {
        for (size_t j = 0; j < order; ++j) {
          indices[j] = (uint8_t**)mxMalloc(sizeof(uint8_t*));
          indices[j][0] = (uint8_t*)mxMalloc(sizeof(int32_t));
          *(int32_t*)(indices[j][0]) = inputs.back().dims[j];
        }
      }
    }

    const std::string op = mxArrayToString(prhs[nrhs - 1]);
    
    bool denseOutput;
    size_t order;
    if (op == "ttv") {
      denseOutput = false;
      order = 2;
    } else if (op == "ttm" || op == "plus") {
      denseOutput = false;
      order = 3;
    } else if (op == "mttkrp") {
      denseOutput = true;
      order = 2;
    } else if (op == "innerprod") {
      order = 0;
      denseOutput = true;
    } else {
      mexErrMsgIdAndTxt("taco:invalidOp", "Invalid operation.");
    }

    // TODO: Initialize output dimensions
    output.order = (int32_t)order;
    output.indices = (uint8_t***)mxMalloc(sizeof(uint8_t**) * order);
    if (denseOutput) {
      for (size_t i = 0; i < order; ++i) {
        output.indices[i] = (uint8_t**)mxMalloc(sizeof(uint8_t*));
        output.indices[i][0] = (uint8_t*)mxMalloc(sizeof(int32_t));
      }
      if (op == "mttkrp") {
        *((int32_t*)output.indices[0][0]) = inputs[0].dims[0];
        *((int32_t*)output.indices[1][0]) = inputs[1].dims[1];
      }
    } else {
      output.indices[0] = (uint8_t**)mxMalloc(sizeof(uint8_t*) * 2);
      for (size_t i = 1; i < order; ++i) {
        output.indices[i] = (uint8_t**)mxMalloc(sizeof(uint8_t*));
      }
    }

    if (op == "ttv") {
      compute_ttv(&output, &inputs[0], &inputs[1]);
    } else if (op == "ttm") {
      compute_ttm(&output, &inputs[0], &inputs[1]);
    } else if (op == "plus") {
      compute_plus(&output, &inputs[0], &inputs[1]);
    } else if (op == "mttkrp") {
      compute_mttkrp(&output, &inputs[0], &inputs[1], &inputs[2]);
    } else if (op == "innerprod") {
      compute_innerprod(&output, &inputs[0], &inputs[1]);
    }

    int nnz = 1;
    if (denseOutput) {
      std::vector<mwSize> dims(order);
      if (order == 0) {
        plhs[0] = mxCreateDoubleScalar(((double*)output.vals)[0]);
      } else {
        for (size_t i = 0; i < order; ++i) {
          dims[i] = (mwSize)(*((int32_t*)output.indices[i][0]));
          nnz *= *((int32_t*)output.indices[i][0]);
        }
        plhs[0] = mxCreateDoubleMatrix(dims[1], dims[0], mxREAL);
      }
    } else {
      nnz = ((int32_t*)(output.indices[0][0]))[1];

      if (nlhs == 2) {
        const mwSize dims[2] = {(mwSize)nnz, (mwSize)order};
        plhs[1] = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);

        memcpy((int32_t*)mxGetData(plhs[1]), (int32_t*)output.indices[0][1], sizeof(int32_t) * nnz);
        free(output.indices[0][1]);
        for (size_t i = 1; i < order; ++i) {
          int32_t* dst = ((int32_t*)mxGetData(plhs[1])) + i * nnz;
          memcpy(dst, (int32_t*)output.indices[i][0], sizeof(int32_t) * nnz);
          free(output.indices[i][0]);
        }
      }

      /* create the output matrix */
      plhs[0] = mxCreateDoubleMatrix((mwSize)nnz, 1, mxREAL);
    }

    if (order > 0) {
      memcpy(mxGetPr(plhs[0]), (double*)output.vals, sizeof(double) * nnz);
    }
    free(output.vals);
}
