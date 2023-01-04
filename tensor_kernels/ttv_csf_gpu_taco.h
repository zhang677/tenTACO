
__global__
void ttv_kernel(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ c, int32_t &fposB0, int32_t* i_blockStarts, int32_t* j_blockStarts){
  float* __restrict__ A_vals = global_A_vals_device_float;
  int B1_dimension = global_B1_dimension;
  int* __restrict__ B2_pos = global_B2_pos_device;
  int* __restrict__ B2_crd = global_B2_crd_device;
  int* __restrict__ B3_pos = global_B3_pos_device;
  int* __restrict__ B3_crd = global_B3_crd_device;
  float* __restrict__ B_vals = global_B_vals_device_float;
  float* __restrict__ c_vals = global_C_vals_device_float;

  int block = blockIdx.x;
  int thread = (threadIdx.x % (32));
  int warp = (threadIdx.x / 32);
  if (threadIdx.x >= 512) {
    return;
  }

  float precomputed[4];
  for (int pprecomputed = 0; pprecomputed < 4; pprecomputed++) {
    precomputed[pprecomputed] = 0.0;
  }
  int thread_nz = 0;
  int fpos2 = thread * 4 + thread_nz;
  int fpos1 = warp * 128 + fpos2;
  int fposB = block * 2048 + fpos1;
  int f = B3_crd[fposB];
  if (block * 2048 + fpos1 + 4 >= B3_pos[B2_pos[B1_dimension]]) {
    for (int thread_nz_pre = 0; thread_nz_pre < 4; thread_nz_pre++) {
      int thread_nz = thread_nz_pre;
      int fpos2 = thread * 4 + thread_nz;
      int fpos1 = warp * 128 + fpos2;
      int fposB = block * 2048 + fpos1;
      if (fposB >= B3_pos[B2_pos[B1_dimension]])
        break;

      int f = B3_crd[fposB];
      precomputed[thread_nz_pre] = B_vals[fposB] * c_vals[f];
    }
  }
  else {
    #pragma unroll 4
    for (int thread_nz_pre = 0; thread_nz_pre < 4; thread_nz_pre++) {
      int thread_nz = thread_nz_pre;
      int fpos2 = thread * 4 + thread_nz;
      int fpos1 = warp * 128 + fpos2;
      int fposB = block * 2048 + fpos1;
      int f = B3_crd[fposB];
      precomputed[thread_nz_pre] = B_vals[fposB] * c_vals[f];
    }
  }
  float tthread_nz_val = 0.0;
  int pB3_begin = j_blockStarts[block];
  int pB3_end = j_blockStarts[(block + 1)];
  int j_pos = taco_binarySearchBefore(B3_pos, pB3_begin, pB3_end, fposB);
  int j = B2_crd[j_pos];
  int pB2_begin = i_blockStarts[block];
  int pB2_end = i_blockStarts[(block + 1)];
  int i_pos = taco_binarySearchBefore(B2_pos, pB2_begin, pB2_end, j_pos);
  int i = i_pos;

  for (int thread_nz = 0; thread_nz < 4; thread_nz++) {
    int fpos2 = thread * 4 + thread_nz;
    int fpos1 = warp * 128 + fpos2;
    int fposB = block * 2048 + fpos1;
    if (fposB >= B3_pos[B2_pos[B1_dimension]])
      break;

    int f = B3_crd[fposB];
    if (fposB == B3_pos[(j_pos + 1)]) {
      j_pos = j_pos + 1;
      j = B2_crd[j_pos];
      while (j_pos == B2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
    }
    tthread_nz_val = tthread_nz_val + precomputed[thread_nz];
    if (fposB + 1 == B3_pos[(j_pos + 1)]) {
      atomicAdd(&A_vals[j_pos], tthread_nz_val); 
      tthread_nz_val = 0.0;
    }
  }
  atomicAddWarp<float>(A_vals, j_pos, tthread_nz_val); 
  
}

void ttv_csf_gpu_taco(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *c) {
  int A1_dimension = (int)(A->dimensions[0]);
  int B1_dimension = (int)(B->dimensions[0]);
  int* __restrict__ B2_pos = (int*)(B->indices[1][0]);
  int* __restrict__ B3_pos = (int*)(B->indices[2][0]);
 copy_to_device_ttv(A, B, c);

  int32_t* j_blockStarts = 0;
  gpuErrchk(cudaMalloc((void**)&j_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_dimension]] + 2047) / 2048 + 1)+1024));
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMalloc((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_dimension]] + 2047) / 2048 + 1)+1024));

  int32_t* fposB_ptr;
  gpuErrchk(cudaMallocManaged((void**)&fposB_ptr, sizeof(int32_t)));
  int32_t& fposB = *fposB_ptr;
  fposB = 0;

TIME_GPU(

  j_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, j_blockStarts, (int32_t) 0, B2_pos[B1_dimension], (int32_t) 2048, (int32_t) 512, ((B3_pos[B2_pos[B1_dimension]] + 2047) / 2048));
  i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_dimension, j_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_dimension]] + 2047) / 2048));
  int32_t status = cudaMemset(global_A_vals_host_copy, 0, A1_dimension * 4);
  ttv_kernel<<<(B3_pos[B2_pos[B1_dimension]] + 2047) / 2048, 32 * 16>>>(A, B, c, fposB, i_blockStarts, j_blockStarts);
);
  cudaDeviceSynchronize();

 copy_from_device_ttv(A, B, c);
 free_tensors_ttv();

 cudaFree(j_blockStarts);
  cudaFree(i_blockStarts);

}

