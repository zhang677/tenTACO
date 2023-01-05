import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2 types codegen')
    parser.add_argument('--alg', default='eb_sr', type=str, help='One of 2 algs')
    parser.add_argument('--first','-f', default=256, type=int, help='First param')
    parser.add_argument('--second','-s', default=16, type=int, help='Second param')
    parser.add_argument('--third','-t', default=4, type=int, help='Third param')
    parser.add_argument('--feat', default=4, type=int, help='Feature dimension')
    parser.add_argument('--tb', default=256, type=int, help='Number of threads per block')
    parser.add_argument('--device','-d', action="store_true", help="Generate device code")
    parser.add_argument('--host', action="store_true", help="Generate host code")
    parser.add_argument('--call', action="store_true", help="Generate callers")
    parser.add_argument('--check', action="store_true", help="Generate checkers")
    parser.add_argument('--tune', default=0, type=int, help="Number of tune")
    args = parser.parse_args()

    feat = args.feat
    threads_per_block = args.tb

    common_prefix = "int A1_dimension = (int)(A->dimensions[0]);\n\
    int A2_dimension = (int)(A->dimensions[1]);\n\
    float* A_vals = (float*)(A->vals);\n\
    int* B1_pos = (int*)(B->indices[0][0]);\n\
    int* B2_pos = (int*)(B->indices[1][0]);\n\
    int* B3_pos = (int*)(B->indices[2][0]);\n\
    copy_to_device_mttkrp(A, B, C, D);\n"

    common_suffix = "copy_from_device_mttkrp(A, B, C, D);\n\
    free_tensors_mttkrp();\n"

    prefix = 'mttkrp3_'+str(threads_per_block)+'_'
    root = '/home/nfs_data/zhanggh/tenTACO/generated_kernels/'
    header = '/home/nfs_data/zhanggh/tenTACO/generated_kernels/fake_head.cuh'
    log_path = root+'fake_ori.h'
    if args.alg == 'eb_sr':
        kernel_name = args.alg+'_'+str(args.first)+'_'+str(args.second)+'_'+str(args.third)
        tune_kernel_name = args.alg + '_tune'
    elif args.alg == 'eb_pr':
        tune_kernel_name = args.alg + '_tune'
    else:
        raise NotImplementedError('Alg not supported!')

    if args.alg == 'eb_sr':
        # g = args.second
        # c = args.third
        warp_size = int(feat / args.third)
        assert warp_size <= threads_per_block, "Warp_size can't be larger than block threads"
        assert args.third <= feat, "c can't be larger than feat"
        nnz_per_block = int(threads_per_block / warp_size * args.second)
        host_name = f"mttkrp_{feat}_{args.alg}_{threads_per_block}_{args.second}_{args.third}"
        device_name = f"{prefix}{tune_kernel_name}"
        with open(log_path, 'a') as f:
            if args.device:
                f.writelines(f"\n\
template<int g, int c>\n\
__global__\n\
void {device_name}(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, taco_tensor_t * __restrict__ D, int32_t* i_blockStarts, int32_t* k_blockStarts){{\n\
  int A2_dimension = global_A2_dimension;\n\
  float* A_vals = global_A_vals_device_float;\n\
  int* B1_pos = global_B1_pos_device;\n\
  int* B1_crd = global_B1_crd_device;\n\
  int* B2_pos = global_B2_pos_device;\n\
  int* B2_crd = global_B2_crd_device;\n\
  int* B3_pos = global_B3_pos_device;\n\
  int* B3_crd = global_B3_crd_device;\n\
  float* B_vals = global_B_vals_device_float;\n\
  int C2_dimension = global_C2_dimension;\n\
  float* C_vals = global_C_vals_device_float;\n\
  int D2_dimension = global_D2_dimension;\n\
  float* D_vals = global_D_vals_device_float;\n\
  int32_t block = blockIdx.x;\n\
  int32_t warp_size = D2_dimension / c;\n\
  int32_t nnz_per_block = {threads_per_block} / warp_size * g;\n\
  int32_t thread = (threadIdx.x % warp_size);\n\
  int32_t warp = (threadIdx.x / warp_size);\n\
  if (threadIdx.x >= {threads_per_block}) {{\n\
    return;\n\
  }}\n\
  for (int32_t dense_val = 0; dense_val < c; dense_val++) {{\n\
    int32_t j = dense_val * warp_size + thread;\n\
    if (j >= D2_dimension)\n\
      break;\n\
    float tnnz_val = 0.0;\n\
    int32_t pB3_begin = k_blockStarts[block];\n\
    int32_t pB3_end = k_blockStarts[(block + 1)];\n\
    int32_t fpos1 = warp * g;\n\
    int32_t fposB = block * nnz_per_block + fpos1;\n\
    if (fposB >= B3_pos[B2_pos[B1_pos[1]]])\n\
      break;\n\
    int32_t k_pos = taco_binarySearchBefore(B3_pos, pB3_begin, pB3_end, fposB);\n\
    int32_t k = B2_crd[k_pos];\n\
    int32_t pB2_begin = i_blockStarts[block];\n\
    int32_t pB2_end = i_blockStarts[(block + 1)];\n\
    int32_t i_pos = taco_binarySearchBefore(B2_pos, pB2_begin, pB2_end, k_pos);\n\
    int32_t i = B1_crd[i_pos];\n\
    for (int32_t nnz = 0; nnz < g; nnz++) {{\n\
      int32_t fpos1 = warp * g + nnz;\n\
      int32_t fposB = block * nnz_per_block + fpos1;\n\
      if (fposB >= B3_pos[B2_pos[B1_pos[1]]])\n\
        break;\n\
      int32_t f = B3_crd[fposB];\n\
      if (fposB == B3_pos[(k_pos + 1)]) {{\n\
        k_pos = k_pos + 1;\n\
        k = B2_crd[k_pos];\n\
        if (k_pos == B2_pos[(i_pos + 1)]) {{\n\
          i_pos = i_pos + 1;\n\
          i = B1_crd[i_pos];\n\
        }}\n\
      }}\n\
        int32_t jA = i * A2_dimension + j;\n\
        int32_t jC = k * C2_dimension + j;\n\
        int32_t jD = f * D2_dimension + j;\n\
        tnnz_val = tnnz_val + B_vals[fposB] * C_vals[jC] * D_vals[jD];\n\
        if (fposB + 1 == B3_pos[(k_pos + 1)]) {{\n\
          atomicAdd(&A_vals[jA], tnnz_val);\n\
          tnnz_val = 0.0;\n\
        }}\n\
    }}\n\
    int32_t jA = i * A2_dimension + j;\n\
    atomicAdd(&A_vals[jA], tnnz_val);\n\
  }}\n\
}}\n")
            if args.host:
                f.writelines(f"\n\
void {host_name}(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {{\n\
    {common_prefix}\n\
    int32_t* k_blockStarts = 0;\n\
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + {nnz_per_block - 1}) / {nnz_per_block} + 1)+1048576));\n\
    int32_t* i_blockStarts = 0;\n\
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + {nnz_per_block - 1}) / {nnz_per_block} + 1)));\n\
    TIME_GPU(\n\
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], {nnz_per_block}, {nnz_per_block}, ((B3_pos[B2_pos[B1_pos[1]]] + {nnz_per_block - 1}) / {nnz_per_block}));\n\
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) {nnz_per_block}, ((B3_pos[B2_pos[B1_pos[1]]] + {nnz_per_block - 1}) / {nnz_per_block}));\n\
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));\n\
        {device_name}<{args.second},{args.third}><<<(B3_pos[B2_pos[B1_pos[1]]] + {nnz_per_block - 1}) / {nnz_per_block}, {threads_per_block}>>>(A, B, C, D, i_blockStarts, k_blockStarts);\n\
    );\n\
    cudaDeviceSynchronize();\n\
    gpuErrchk(cudaGetLastError());\n\
    cudaFree(k_blockStarts);\n\
    cudaFree(i_blockStarts);\n\
    {common_suffix}\n\
}}\n") 

    elif args.alg == 'eb_pr':
        # c = args.second
        # group_size = args.third
        assert args.second <= feat, "Nnz can't exceed threads_per_block"
        Nnz = int(threads_per_block / (feat / args.second))
        assert args.third <= Nnz, f"Group size {args.third} can't exceed warp size {Nnz}"
        device_name = f"{prefix}{tune_kernel_name}"
        host_name = f"mttkrp_{feat}_{args.alg}_{threads_per_block}_{Nnz}_{args.third}"
        with open(log_path, 'a') as f:
            if args.device:
                f.writelines(f"\n\
template<int Nnz, int c, int group_size>\n\
__global__\n\
void {device_name}(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, taco_tensor_t * __restrict__ D, int32_t* i_blockStarts, int32_t* k_blockStarts){{\n\
  int A2_dimension = global_A2_dimension;\n\
  float* A_vals = global_A_vals_device_float;\n\
  int* B1_pos = global_B1_pos_device;\n\
  int* B1_crd = global_B1_crd_device;\n\
  int* B2_pos = global_B2_pos_device;\n\
  int* B2_crd = global_B2_crd_device;\n\
  int* B3_pos = global_B3_pos_device;\n\
  int* B3_crd = global_B3_crd_device;\n\
  float* B_vals = global_B_vals_device_float;\n\
  int C2_dimension = global_C2_dimension;\n\
  float* C_vals = global_C_vals_device_float;\n\
  int D2_dimension = global_D2_dimension;\n\
  float* D_vals = global_D_vals_device_float;\n\
  int32_t block = blockIdx.x;\n\
  int32_t fpos1 = threadIdx.x % Nnz;\n\
  int32_t jo = threadIdx.x / Nnz;\n\
  if (threadIdx.x >= {threads_per_block}) {{\n\
      return;\n\
  }}\n\
  for (int32_t ji = 0; ji < c; ji++) {{\n\
    int32_t j = jo * c + ji;\n\
    if (j >= D2_dimension)\n\
      break;\n\
    int32_t pB3_begin = k_blockStarts[block];\n\
    int32_t pB3_end = k_blockStarts[(block + 1)];\n\
    int32_t fposB = block * Nnz + fpos1;\n\
    int32_t k_pos = taco_binarySearchBefore(B3_pos, pB3_begin, pB3_end, fposB);\n\
    int32_t k = B2_crd[k_pos];\n\
    int32_t pB2_begin = i_blockStarts[block];\n\
    int32_t pB2_end = i_blockStarts[(block + 1)];\n\
    int32_t i_pos = taco_binarySearchBefore(B2_pos, pB2_begin, pB2_end, k_pos);\n\
    int32_t i = B1_crd[i_pos];\n\
    float tnnz_val = 0.0;\n\
    if (fposB >= B3_pos[B2_pos[B1_pos[1]]]) tnnz_val = 0.0;\n\
    else {{\n\
      int32_t fposB = block * Nnz + fpos1;\n\
      int32_t f = B3_crd[fposB];\n\
      int32_t jC = k * C2_dimension + j;\n\
      int32_t jD = f * D2_dimension + j;\n\
      if (fposB == B3_pos[(k_pos + 1)]) {{\n\
        k_pos = k_pos + 1;\n\
        k = B2_crd[k_pos];\n\
        if (k_pos == B2_pos[(i_pos + 1)]) {{\n\
          i_pos = i_pos + 1;\n\
          i = B1_crd[i_pos];\n\
        }}\n\
      }}\n\
      tnnz_val = B_vals[fposB] * C_vals[jC] * D_vals[jD];\n\
    }}\n\
    int32_t jA = i * A2_dimension + j;\n\
    segReduceGroup<float, group_size>(A_vals, jA, tnnz_val);\n\
  }}\n\
}}\n")
            if args.host:
                f.writelines(f"\n\
void {host_name}(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {{\n\
    {common_prefix}\n\
    int32_t* k_blockStarts = 0;\n\
    gpuErrchk(cudaMallocManaged((void**)&k_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + {Nnz - 1}) / {Nnz} + 1)+1048576));\n\
    int32_t* i_blockStarts = 0;\n\
    gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B3_pos[B2_pos[B1_pos[1]]] + {Nnz - 1}) / {Nnz} + 1)));\n\
    TIME_GPU(\n\
        k_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B3_pos_host_copy, k_blockStarts, 0, B2_pos[B1_pos[1]], {Nnz}, {Nnz}, ((B3_pos[B2_pos[B1_pos[1]]] + {Nnz - 1}) / {Nnz}));\n\
        i_blockStarts = taco_binarySearchIndirectBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_pos[1], k_blockStarts, (int32_t) {Nnz}, ((B3_pos[B2_pos[B1_pos[1]]] + {Nnz - 1}) / {Nnz}));\n\
        gpuErrchk(cudaMemset(global_A_vals_host_copy_float, 0, ((size_t) A1_dimension * A2_dimension) * 4));\n\
        {device_name}<{Nnz},{args.second},{args.third}><<<(B3_pos[B2_pos[B1_pos[1]]] + {Nnz - 1}) / {Nnz}, {threads_per_block}>>>(A, B, C, D, i_blockStarts, k_blockStarts);\n\
    );\n\
    cudaDeviceSynchronize();\n\
    gpuErrchk(cudaGetLastError());\n\
    cudaFree(k_blockStarts);\n\
    cudaFree(i_blockStarts);\n\
    {common_suffix}\n\
}}\n")

    with open(header, 'r') as f:
        lines = f.readlines()
        index = lines.index('#endif\n')
        lines.append('#endif\n')
        lines[index] = f"void {host_name}(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D);\n"
    with open(header, 'w') as f:
        f.writelines(lines)
    if args.check:
        with open("checkers.txt", "a") as f:
            f.write(f"{host_name}(&mats[0], &B_taco, &mats[1], &mats[2]);std::cout << \"taco CPU vs taco GPU: \" << compare_matrices_float(mats_ref[0], mats[0]) << std::endl;\n")
    
    if args.call:
        names = (args.alg).split('_')
        with open("callers.txt", "a") as f:
            f.write(f"RUN_GPU( {host_name}(&mats[0], &B_taco, &mats[1], &mats[2]);, trials,\"mttkrp\", \"{names[0]}\", \"{names[1]}\", \"tune{args.tune}\", tensor_name);\n")