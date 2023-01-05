
template<int Nnz, int c, int group_size>
__global__
void mttkrp3_32_eb_pr_tune(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, taco_tensor_t * __restrict__ D, int32_t* i_blockStarts, int32_t* k_blockStarts){
  int A2_dimension = global_A2_dimension;
  float* A_vals = global_A_vals_device_float;
  int* B1_pos = global_B1_pos_device;
  int* B1_crd = global_B1_crd_device;
  int* B2_pos = global_B2_pos_device;
  int* B2_crd = global_B2_crd_device;
  int* B3_pos = global_B3_pos_device;
  int* B3_crd = global_B3_crd_device;
  float* B_vals = global_B_vals_device_float;
  int C2_dimension = global_C2_dimension;
  float* C_vals = global_C_vals_device_float;
  int D2_dimension = global_D2_dimension;
  float* D_vals = global_D_vals_device_float;
  int32_t block = blockIdx.x;
  int32_t fpos1 = threadIdx.x % Nnz;
  int32_t jo = threadIdx.x / Nnz;
  if (threadIdx.x >= 256) {
      return;
  }
  for (int32_t ji = 0; ji < c; ji++) {
    int32_t j = jo * c + ji;
    if (j >= D2_dimension)
      break;
    int32_t pB3_begin = k_blockStarts[block];
    int32_t pB3_end = k_blockStarts[(block + 1)];
    int32_t fposB = block * Nnz + fpos1;
    int32_t k_pos = taco_binarySearchBefore(B3_pos, pB3_begin, pB3_end, fposB);
    int32_t k = B2_crd[k_pos];
    int32_t pB2_begin = i_blockStarts[block];
    int32_t pB2_end = i_blockStarts[(block + 1)];
    int32_t i_pos = taco_binarySearchBefore(B2_pos, pB2_begin, pB2_end, k_pos);
    int32_t i = B1_crd[i_pos];
    float tnnz_val = 0.0;
    if (fposB >= B3_pos[B2_pos[B1_pos[1]]]) tnnz_val = 0.0;
    else {
      int32_t fposB = block * Nnz + fpos1;
      int32_t f = B3_crd[fposB];
      int32_t jC = k * C2_dimension + j;
      int32_t jD = f * D2_dimension + j;
      if (fposB == B3_pos[(k_pos + 1)]) {
        k_pos = k_pos + 1;
        k = B2_crd[k_pos];
        if (k_pos == B2_pos[(i_pos + 1)]) {
          i_pos = i_pos + 1;
          i = B1_crd[i_pos];
        }
      }
      tnnz_val = B_vals[fposB] * C_vals[jC] * D_vals[jD];
    }
    int32_t jA = i * A2_dimension + j;
    segReduceGroup<float, group_size>(A_vals, jA, tnnz_val);
  }
}

void spmm_32_eb_pr_256_64_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_32_eb_pr_tune<64,8,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}


template<int Nnz, int c, int group_size>
__global__
void mttkrp3_64_eb_pr_tune(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, taco_tensor_t * __restrict__ D, int32_t* i_blockStarts, int32_t* k_blockStarts){
  int A2_dimension = global_A2_dimension;
  float* A_vals = global_A_vals_device_float;
  int* B1_pos = global_B1_pos_device;
  int* B1_crd = global_B1_crd_device;
  int* B2_pos = global_B2_pos_device;
  int* B2_crd = global_B2_crd_device;
  int* B3_pos = global_B3_pos_device;
  int* B3_crd = global_B3_crd_device;
  float* B_vals = global_B_vals_device_float;
  int C2_dimension = global_C2_dimension;
  float* C_vals = global_C_vals_device_float;
  int D2_dimension = global_D2_dimension;
  float* D_vals = global_D_vals_device_float;
  int32_t block = blockIdx.x;
  int32_t fpos1 = threadIdx.x % Nnz;
  int32_t jo = threadIdx.x / Nnz;
  if (threadIdx.x >= 256) {
      return;
  }
  for (int32_t ji = 0; ji < c; ji++) {
    int32_t j = jo * c + ji;
    if (j >= D2_dimension)
      break;
    int32_t pB3_begin = k_blockStarts[block];
    int32_t pB3_end = k_blockStarts[(block + 1)];
    int32_t fposB = block * Nnz + fpos1;
    int32_t k_pos = taco_binarySearchBefore(B3_pos, pB3_begin, pB3_end, fposB);
    int32_t k = B2_crd[k_pos];
    int32_t pB2_begin = i_blockStarts[block];
    int32_t pB2_end = i_blockStarts[(block + 1)];
    int32_t i_pos = taco_binarySearchBefore(B2_pos, pB2_begin, pB2_end, k_pos);
    int32_t i = B1_crd[i_pos];
    float tnnz_val = 0.0;
    if (fposB >= B3_pos[B2_pos[B1_pos[1]]]) tnnz_val = 0.0;
    else {
      int32_t fposB = block * Nnz + fpos1;
      int32_t f = B3_crd[fposB];
      int32_t jC = k * C2_dimension + j;
      int32_t jD = f * D2_dimension + j;
      if (fposB == B3_pos[(k_pos + 1)]) {
        k_pos = k_pos + 1;
        k = B2_crd[k_pos];
        if (k_pos == B2_pos[(i_pos + 1)]) {
          i_pos = i_pos + 1;
          i = B1_crd[i_pos];
        }
      }
      tnnz_val = B_vals[fposB] * C_vals[jC] * D_vals[jD];
    }
    int32_t jA = i * A2_dimension + j;
    segReduceGroup<float, group_size>(A_vals, jA, tnnz_val);
  }
}

void mttkrp_64_eb_pr_256_32_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_64_eb_pr_tune<32,8,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 31) / 32, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

template<int Nnz, int c, int group_size>
__global__
void mttkrp3_32_eb_pr_tune(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, taco_tensor_t * __restrict__ D, int32_t* i_blockStarts, int32_t* k_blockStarts){
  int A2_dimension = global_A2_dimension;
  float* A_vals = global_A_vals_device_float;
  int* B1_pos = global_B1_pos_device;
  int* B1_crd = global_B1_crd_device;
  int* B2_pos = global_B2_pos_device;
  int* B2_crd = global_B2_crd_device;
  int* B3_pos = global_B3_pos_device;
  int* B3_crd = global_B3_crd_device;
  float* B_vals = global_B_vals_device_float;
  int C2_dimension = global_C2_dimension;
  float* C_vals = global_C_vals_device_float;
  int D2_dimension = global_D2_dimension;
  float* D_vals = global_D_vals_device_float;
  int32_t block = blockIdx.x;
  int32_t fpos1 = threadIdx.x % Nnz;
  int32_t jo = threadIdx.x / Nnz;
  if (threadIdx.x >= 256) {
      return;
  }
  for (int32_t ji = 0; ji < c; ji++) {
    int32_t j = jo * c + ji;
    if (j >= D2_dimension)
      break;
    int32_t pB3_begin = k_blockStarts[block];
    int32_t pB3_end = k_blockStarts[(block + 1)];
    int32_t fposB = block * Nnz + fpos1;
    int32_t k_pos = taco_binarySearchBefore(B3_pos, pB3_begin, pB3_end, fposB);
    int32_t k = B2_crd[k_pos];
    int32_t pB2_begin = i_blockStarts[block];
    int32_t pB2_end = i_blockStarts[(block + 1)];
    int32_t i_pos = taco_binarySearchBefore(B2_pos, pB2_begin, pB2_end, k_pos);
    int32_t i = B1_crd[i_pos];
    float tnnz_val = 0.0;
    if (fposB >= B3_pos[B2_pos[B1_pos[1]]]) tnnz_val = 0.0;
    else {
      int32_t fposB = block * Nnz + fpos1;
      int32_t f = B3_crd[fposB];
      int32_t jC = k * C2_dimension + j;
      int32_t jD = f * D2_dimension + j;
      if (fposB == B3_pos[(k_pos + 1)]) {
        k_pos = k_pos + 1;
        k = B2_crd[k_pos];
        if (k_pos == B2_pos[(i_pos + 1)]) {
          i_pos = i_pos + 1;
          i = B1_crd[i_pos];
        }
      }
      tnnz_val = B_vals[fposB] * C_vals[jC] * D_vals[jD];
    }
    int32_t jA = i * A2_dimension + j;
    segReduceGroup<float, group_size>(A_vals, jA, tnnz_val);
  }
}

void mttkrp_32_eb_pr_256_256_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_32_eb_pr_tune<256,32,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

template<int Nnz, int c, int group_size>
__global__
void mttkrp3_32_eb_pr_tune(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, taco_tensor_t * __restrict__ D, int32_t* i_blockStarts, int32_t* k_blockStarts){
  int A2_dimension = global_A2_dimension;
  float* A_vals = global_A_vals_device_float;
  int* B1_pos = global_B1_pos_device;
  int* B1_crd = global_B1_crd_device;
  int* B2_pos = global_B2_pos_device;
  int* B2_crd = global_B2_crd_device;
  int* B3_pos = global_B3_pos_device;
  int* B3_crd = global_B3_crd_device;
  float* B_vals = global_B_vals_device_float;
  int C2_dimension = global_C2_dimension;
  float* C_vals = global_C_vals_device_float;
  int D2_dimension = global_D2_dimension;
  float* D_vals = global_D_vals_device_float;
  int32_t block = blockIdx.x;
  int32_t fpos1 = threadIdx.x % Nnz;
  int32_t jo = threadIdx.x / Nnz;
  if (threadIdx.x >= 256) {
      return;
  }
  for (int32_t ji = 0; ji < c; ji++) {
    int32_t j = jo * c + ji;
    if (j >= D2_dimension)
      break;
    int32_t pB3_begin = k_blockStarts[block];
    int32_t pB3_end = k_blockStarts[(block + 1)];
    int32_t fposB = block * Nnz + fpos1;
    int32_t k_pos = taco_binarySearchBefore(B3_pos, pB3_begin, pB3_end, fposB);
    int32_t k = B2_crd[k_pos];
    int32_t pB2_begin = i_blockStarts[block];
    int32_t pB2_end = i_blockStarts[(block + 1)];
    int32_t i_pos = taco_binarySearchBefore(B2_pos, pB2_begin, pB2_end, k_pos);
    int32_t i = B1_crd[i_pos];
    float tnnz_val = 0.0;
    if (fposB >= B3_pos[B2_pos[B1_pos[1]]]) tnnz_val = 0.0;
    else {
      int32_t fposB = block * Nnz + fpos1;
      int32_t f = B3_crd[fposB];
      int32_t jC = k * C2_dimension + j;
      int32_t jD = f * D2_dimension + j;
      if (fposB == B3_pos[(k_pos + 1)]) {
        k_pos = k_pos + 1;
        k = B2_crd[k_pos];
        if (k_pos == B2_pos[(i_pos + 1)]) {
          i_pos = i_pos + 1;
          i = B1_crd[i_pos];
        }
      }
      tnnz_val = B_vals[fposB] * C_vals[jC] * D_vals[jD];
    }
    int32_t jA = i * A2_dimension + j;
    segReduceGroup<float, group_size>(A_vals, jA, tnnz_val);
  }
}

void mttkrp_32_eb_pr_256_256_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_32_eb_pr_tune<256,32,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

template<int Nnz, int c, int group_size>
__global__
void mttkrp3_32_eb_pr_tune(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, taco_tensor_t * __restrict__ D, int32_t* i_blockStarts, int32_t* k_blockStarts){
  int A2_dimension = global_A2_dimension;
  float* A_vals = global_A_vals_device_float;
  int* B1_pos = global_B1_pos_device;
  int* B1_crd = global_B1_crd_device;
  int* B2_pos = global_B2_pos_device;
  int* B2_crd = global_B2_crd_device;
  int* B3_pos = global_B3_pos_device;
  int* B3_crd = global_B3_crd_device;
  float* B_vals = global_B_vals_device_float;
  int C2_dimension = global_C2_dimension;
  float* C_vals = global_C_vals_device_float;
  int D2_dimension = global_D2_dimension;
  float* D_vals = global_D_vals_device_float;
  int32_t block = blockIdx.x;
  int32_t fpos1 = threadIdx.x % Nnz;
  int32_t jo = threadIdx.x / Nnz;
  if (threadIdx.x >= 256) {
      return;
  }
  for (int32_t ji = 0; ji < c; ji++) {
    int32_t j = jo * c + ji;
    if (j >= D2_dimension)
      break;
    int32_t pB3_begin = k_blockStarts[block];
    int32_t pB3_end = k_blockStarts[(block + 1)];
    int32_t fposB = block * Nnz + fpos1;
    int32_t k_pos = taco_binarySearchBefore(B3_pos, pB3_begin, pB3_end, fposB);
    int32_t k = B2_crd[k_pos];
    int32_t pB2_begin = i_blockStarts[block];
    int32_t pB2_end = i_blockStarts[(block + 1)];
    int32_t i_pos = taco_binarySearchBefore(B2_pos, pB2_begin, pB2_end, k_pos);
    int32_t i = B1_crd[i_pos];
    float tnnz_val = 0.0;
    if (fposB >= B3_pos[B2_pos[B1_pos[1]]]) tnnz_val = 0.0;
    else {
      int32_t fposB = block * Nnz + fpos1;
      int32_t f = B3_crd[fposB];
      int32_t jC = k * C2_dimension + j;
      int32_t jD = f * D2_dimension + j;
      if (fposB == B3_pos[(k_pos + 1)]) {
        k_pos = k_pos + 1;
        k = B2_crd[k_pos];
        if (k_pos == B2_pos[(i_pos + 1)]) {
          i_pos = i_pos + 1;
          i = B1_crd[i_pos];
        }
      }
      tnnz_val = B_vals[fposB] * C_vals[jC] * D_vals[jD];
    }
    int32_t jA = i * A2_dimension + j;
    segReduceGroup<float, group_size>(A_vals, jA, tnnz_val);
  }
}

void mttkrp_32_eb_pr_256_64_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_32_eb_pr_tune<64,8,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_32_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 32, 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<32,4,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 31) / 32, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_32_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 32, 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<32,4,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 31) / 32, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_32_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 32, 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<32,4,16><<<(B3_pos[B2_pos[B1_pos[1]]] + 31) / 32, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_32_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 32, 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<32,4,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 31) / 32, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_64_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<64,8,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_64_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<64,8,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_64_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<64,8,16><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_64_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<64,8,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_128_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<128,16,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_128_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<128,16,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_128_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<128,16,16><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_128_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<128,16,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_32_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 32, 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<32,4,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 31) / 32, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_32_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 32, 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<32,4,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 31) / 32, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_32_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 32, 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<32,4,16><<<(B3_pos[B2_pos[B1_pos[1]]] + 31) / 32, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_32_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 32, 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<32,4,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 31) / 32, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_64_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<64,8,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_64_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<64,8,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_64_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<64,8,16><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_64_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<64,8,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_128_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<128,16,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_128_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<128,16,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_128_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<128,16,16><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_128_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<128,16,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_256_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<256,32,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_256_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<256,32,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_256_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<256,32,16><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_256_256_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_eb_pr_tune<256,32,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_64_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<64,4,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_64_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<64,4,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_64_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<64,4,16><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_64_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<64,4,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_128_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<128,8,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_128_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<128,8,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_128_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<128,8,16><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_128_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<128,8,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_256_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<256,16,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_256_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<256,16,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_256_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<256,16,16><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_256_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<256,16,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_512_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 512, 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<512,32,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 511) / 512, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_512_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 512, 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<512,32,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 511) / 512, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_512_16(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 512, 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<512,32,16><<<(B3_pos[B2_pos[B1_pos[1]]] + 511) / 512, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_pr_512_512_32(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 512, 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_pr_tune<512,32,32><<<(B3_pos[B2_pos[B1_pos[1]]] + 511) / 512, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_4_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 32, 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 32, ((B3_pos[B2_pos[B1_pos[1]]] + 31) / 32));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<4,1><<<(B3_pos[B2_pos[B1_pos[1]]] + 31) / 32, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_4_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<4,2><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_4_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<4,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_4_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<4,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_8_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<8,1><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_8_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<8,2><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_8_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<8,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_8_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 512, 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<8,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 511) / 512, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_16_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<16,1><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_16_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<16,2><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_16_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 512, 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<16,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 511) / 512, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_16_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 1024, 1024, ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 1024, ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<16,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_32_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<32,1><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_32_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 512, 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<32,2><<<(B3_pos[B2_pos[B1_pos[1]]] + 511) / 512, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_32_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 1024, 1024, ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 1024, ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<32,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_256_32_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 2048, 2048, ((B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 2048, ((B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_256_eb_sr_tune<32,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048, 256>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_4_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 64, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<4,1><<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_4_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<4,2><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_4_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<4,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_4_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 512, 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<4,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 511) / 512, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_8_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 128, 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 128, ((B3_pos[B2_pos[B1_pos[1]]] + 127) / 128));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<8,1><<<(B3_pos[B2_pos[B1_pos[1]]] + 127) / 128, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_8_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<8,2><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_8_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 512, 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<8,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 511) / 512, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_8_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 1024, 1024, ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 1024, ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<8,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_16_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 256, 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 256, ((B3_pos[B2_pos[B1_pos[1]]] + 255) / 256));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<16,1><<<(B3_pos[B2_pos[B1_pos[1]]] + 255) / 256, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_16_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 512, 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<16,2><<<(B3_pos[B2_pos[B1_pos[1]]] + 511) / 512, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_16_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 1024, 1024, ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 1024, ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<16,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_16_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 2048, 2048, ((B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 2048, ((B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<16,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_32_1(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 512, 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_pos[1]]] + 511) / 512));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<32,1><<<(B3_pos[B2_pos[B1_pos[1]]] + 511) / 512, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_32_2(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 1024, 1024, ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 1024, ((B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<32,2><<<(B3_pos[B2_pos[B1_pos[1]]] + 1023) / 1024, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_32_4(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 2048, 2048, ((B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 2048, ((B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<32,4><<<(B3_pos[B2_pos[B1_pos[1]]] + 2047) / 2048, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}

void mttkrp_32_eb_sr_512_32_8(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    float* A_vals = (float*)(A->vals);
    int* B1_pos = (int*)(B->indices[0][0]);
    int* B2_pos = (int*)(B->indices[1][0]);
    int* B3_pos = (int*)(B->indices[2][0]);
    copy_to_device_mttkrp(A, B, C, D);

    int32_t* k_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 4095) / 4096 + 1)+1048576));
    int32_t* i_blockStarts = 0;
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + 4095) / 4096 + 1)));
    TIME_GPU(
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 4096, 4096, ((B3_pos[B2_pos[B1_pos[1]]] + 4095) / 4096));
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 4096, ((B3_pos[B2_pos[B1_pos[1]]] + 4095) / 4096));
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
        mttkrp3_512_eb_sr_tune<32,8><<<(B3_pos[B2_pos[B1_pos[1]]] + 4095) / 4096, 512>>>(A, B, C, D, i_blockStarts, k_blockStarts);
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(k_blockStarts);
    cudaFree(i_blockStarts);
    copy_from_device_mttkrp(A, B, C, D);
    free_tensors_mttkrp();

}
