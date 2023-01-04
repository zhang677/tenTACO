int compute_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *c) {
  double* restrict A_vals = (double*)(A->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int* restrict B2_pos = (int*)(B->indices[1][0]);
  int* restrict B2_crd = (int*)(B->indices[1][1]);
  int* restrict B3_pos = (int*)(B->indices[2][0]);
  int* restrict B3_crd = (int*)(B->indices[2][1]);
  double* restrict B_vals = (double*)(B->vals);
  int c1_dimension = (int)(c->dimensions[0]);
  double* restrict c_vals = (double*)(c->vals);

  int32_t jB = 0;

  for (int32_t i = 0; i < B1_dimension; i++) {
    for (int32_t jB = B2_pos[i]; jB < B2_pos[(i + 1)]; jB++) {
      for (int32_t kB = B3_pos[jB]; kB < B3_pos[(jB + 1)]; kB++) {
        int32_t k = B3_crd[kB];
        A_vals[jB] = A_vals[jB] + B_vals[kB] * c_vals[k];
      }
    }
  }
  return 0;
}

int compute_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *c) {
  double* restrict A_vals = (double*)(A->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int* restrict B2_pos = (int*)(B->indices[1][0]);
  int* restrict B2_crd = (int*)(B->indices[1][1]);
  int* restrict B3_pos = (int*)(B->indices[2][0]);
  int* restrict B3_crd = (int*)(B->indices[2][1]);
  double* restrict B_vals = (double*)(B->vals);
  int c1_dimension = (int)(c->dimensions[0]);
  double* restrict c_vals = (double*)(c->vals);

  int32_t jB = 0;

  for (int32_t i = 0; i < B1_dimension; i++) {
    for (int32_t jB = B2_pos[i]; jB < B2_pos[(i + 1)]; jB++) {
      for (int32_t kB = B3_pos[jB]; kB < B3_pos[(jB + 1)]; kB++) {
        int32_t k = B3_crd[kB];
        A_vals[jB] = A_vals[jB] + B_vals[kB] * c_vals[k];
      }
    }
  }
  return 0;
}

void ttv_csf_cpu_taco_unscheduled(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *c) {
TIME_COLD(
  double* restrict A_vals = (double*)(A->vals);
  long B1_dimension = (long)(B->dimensions[0]);
  long* restrict B2_pos = (long*)(B->indices[1][0]);
  long* restrict B2_crd = (long*)(B->indices[1][1]);
  long* restrict B3_pos = (long*)(B->indices[2][0]);
  long* restrict B3_crd = (long*)(B->indices[2][1]);
  double* restrict B_vals = (double*)(B->vals);
  long c1_dimension = (long)(c->dimensions[0]);
  double* restrict c_vals = (double*)(c->vals);
  long a1_dimension = (long)(A->dimensions[0]);

  long jB = 0;

    _Pragma("omp parallel for schedule(static)")
  for (long py = 0; py < a1_dimension; py++) {
    A_vals[py] = 0.0;
  }

  _Pragma("omp parallel for")
  for (long i = 0; i < B1_dimension; i++) {
    for (long jB = B2_pos[i]; jB < B2_pos[(i + 1)]; jB++) {
      for (long kB = B3_pos[jB]; kB < B3_pos[(jB + 1)]; kB++) {
        long k = B3_crd[kB];
        A_vals[jB] = A_vals[jB] + B_vals[kB] * c_vals[k];
      }
    }
  }
  );
}
