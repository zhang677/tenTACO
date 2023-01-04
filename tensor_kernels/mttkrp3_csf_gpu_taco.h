#include "gpu_library.h"
#include "gpu_kernels.cuh"

__global__
void mttkrp_gpu_taco_kernel(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, taco_tensor_t * __restrict__ D, int32_t* i_blockStarts, int32_t* k_blockStarts){
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
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 512) {
    return;
  }

  for (int32_t dense_val = 0; dense_val < 1; dense_val++) {
    int32_t j = dense_val * 32 + thread;
    float tnnz_val = 0.0;
    int32_t pB3_begin = k_blockStarts[block];
    int32_t pB3_end = k_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 4;
    int32_t fposB = block * 64 + fpos1;
    if (fposB >= B3_pos[B2_pos[B1_pos[1]]])
      break;
  
    int32_t k_pos = taco_binarySearchBefore(B3_pos, pB3_begin, pB3_end, fposB);
    int32_t k = B2_crd[k_pos];
    int32_t pB2_begin = i_blockStarts[block];
    int32_t pB2_end = i_blockStarts[(block + 1)];
    int32_t i_pos = taco_binarySearchBefore(B2_pos, pB2_begin, pB2_end, k_pos);
    int32_t i = B1_crd[i_pos];
    for (int32_t nnz = 0; nnz < 4; nnz++) {
      int32_t fpos1 = warp * 4 + nnz;
      int32_t fposB = block * 64 + fpos1;
      if (fposB >= B3_pos[B2_pos[B1_pos[1]]])
        break;

      int32_t f = B3_crd[fposB];
      if (fposB == B3_pos[(k_pos + 1)]) {
        k_pos = k_pos + 1;
        k = B2_crd[k_pos];
        if (k_pos == B2_pos[(i_pos + 1)]) {
          i_pos = i_pos + 1;
          i = B1_crd[i_pos];
        }
      }
      int32_t jA = i * A2_dimension + j;
      int32_t jC = k * C2_dimension + j;
      int32_t jD = f * D2_dimension + j;
      tnnz_val = tnnz_val + B_vals[fposB] * C_vals[jC] * D_vals[jD];

      if (fposB + 1 == B3_pos[(k_pos + 1)]) {
        atomicAdd(&A_vals[jA], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    int32_t jA = i * A2_dimension + j;
    atomicAdd(&A_vals[jA], tnnz_val);
  }
}

void mttkrp_gpu_taco(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
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
  k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], 64, 512, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
  i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) 512, ((B3_pos[B2_pos[B1_pos[1]]] + 63) / 64));
  
  gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));
  mttkrp_gpu_taco_kernel<<<(B3_pos[B2_pos[B1_pos[1]]] + 63) / 64, 32 * 16>>>(A, B, C, D, i_blockStarts, k_blockStarts);
);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(k_blockStarts);
  cudaFree(i_blockStarts);
  copy_from_device_mttkrp(A, B, C, D);
  free_tensors_mttkrp();
  
}
