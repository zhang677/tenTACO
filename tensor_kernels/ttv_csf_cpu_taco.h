
long taco_binarySearchBefore(long *array, long arrayStart, long arrayEnd, long target) {
  if (array[arrayEnd] <= target) {
    return arrayEnd;
  }
  long lowerBound = arrayStart; // always <= target
  long upperBound = arrayEnd; // always > target
  while (upperBound - lowerBound > 1) {
    long mid = (upperBound + lowerBound) / 2;
    long midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return lowerBound;
}

void ttv_csf_cpu_taco(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *c) {
TIME_COLD(
  double* restrict A_vals = (double*)(A->vals);
  long B1_dimension = (long)(B->dimensions[0]);
  long B2_dimension = (long)(B->dimensions[1]);
  long* restrict B2_pos = (long*)(B->indices[1][0]);
  long* restrict B2_crd = (long*)(B->indices[1][1]);
  long* restrict B3_pos = (long*)(B->indices[2][0]);
  long* restrict B3_crd = (long*)(B->indices[2][1]);
  double* restrict B_vals = (double*)(B->vals);
  long c1_dimension = (long)(c->dimensions[0]);
  long a1_dimension = (long)(A->dimensions[0]);
  double* restrict c_vals = (double*)(c->vals);

  long pB2_begin = 0;
  long pB2_end = B1_dimension;

  _Pragma("omp parallel for schedule(static)")
  for (long py = 0; py < a1_dimension; py++) {
    A_vals[py] = 0.0;
  }

  _Pragma("omp parallel for schedule(dynamic, 1)")
  for (long chunk = 0; chunk < ((B2_pos[B1_dimension] + 7) / 8); chunk++) {
    long fposB = chunk * 8;
    long i_pos = taco_binarySearchBefore(B2_pos, pB2_begin, pB2_end, fposB);
    long i = i_pos;
    for (long fpos2 = 0; fpos2 < 8; fpos2++) {
      long fposB = chunk * 8 + fpos2;
      if (fposB >= B2_pos[B1_dimension])
        continue;

      long f = B2_crd[fposB];
      while (fposB == B2_pos[(i_pos + 1)]) {
        i_pos++;
        i = i_pos;
      }
      for (long kB = B3_pos[fposB]; kB < B3_pos[(fposB + 1)]; kB++) {
        long k = B3_crd[kB];
        A_vals[fposB] = A_vals[fposB] + B_vals[kB] * c_vals[k];
      }
    }
  }
);
}


