#ifndef MTTKRP_CSF_CPU_H
#define MTTKRP_CSF_CPU_H

#include "ds.h"
#include "timers.h"

#include "tensor_kernels/mttkrp_csf_cpu_splatt.h"
#include "tensor_kernels/mttkrp3_csf_cpu_taco_unscheduled.h"
#include "tensor_kernels/mttkrp3_csf_cpu_taco.h"
#include "tensor_kernels/mttkrp3_csf_cpu_taco_workspace.h"
#include "tensor_kernels/mttkrp4_csf_cpu_taco_unscheduled.h"
#include "tensor_kernels/mttkrp4_csf_cpu_taco.h"
#include "tensor_kernels/mttkrp4_csf_cpu_taco_workspace.h"
#include "tensor_kernels/mttkrp5_csf_cpu_taco_unscheduled.h"
#include "tensor_kernels/mttkrp5_csf_cpu_taco.h"
#include "tensor_kernels/mttkrp5_csf_cpu_taco_workspace.h"

void mttkrp_csf_cpu(splatt_csf* B_splatt, const std::string& tensor_name, const bool do_verify, int num_cols) {
  taco_tensor_t B_taco = to_taco_tensor(B_splatt);
  
  int J = num_cols;

  splatt_kruskal factor_matrices = gen_factor_matrices(J, B_splatt);
  std::vector<taco_tensor_t> mats(B_splatt->nmodes);
  for (int i = 0; i < B_splatt->nmodes; ++i) {
    mats[i] = to_taco_tensor(factor_matrices, B_splatt, i);
  }

  if (do_verify) {
    splatt_kruskal factor_matrices_ref = gen_factor_matrices(J, B_splatt);
    for (int i = 0; i < B_splatt->nmodes; ++i) {
      if (i != B_splatt->dim_perm[0]) {
        factor_matrices_ref.factors[i] = factor_matrices.factors[i];
      }
    }
    taco_tensor_t Aref_taco = to_taco_tensor(factor_matrices_ref, B_splatt, 0);

    mttkrp_cpu_splatt(B_splatt, &factor_matrices_ref);
    switch (B_splatt->nmodes) {
      case 3:
        mttkrp3_cpu_taco_unscheduled(&mats[0], &B_taco, &mats[1], &mats[2]);
        break;
      case 4:
        mttkrp4_cpu_taco_unscheduled(&mats[0], &B_taco, &mats[1], &mats[2], &mats[3]);
        break;
      case 5:
        mttkrp5_cpu_taco_unscheduled(&mats[0], &B_taco, &mats[1], &mats[2], &mats[3], &mats[4]);
        break;
      default:
        std::cout << "Unsupported number of modes!" << std::endl;
        exit(1);
        break;
    }
    std::cout << "taco unscheduled vs splatt: " << compare_matrices(Aref_taco, mats[0]) << std::endl;

    switch (B_splatt->nmodes) {
      case 3:
        mttkrp3_cpu_taco(&mats[0], &B_taco, &mats[1], &mats[2]);
        break;
      case 4:
        mttkrp4_cpu_taco(&mats[0], &B_taco, &mats[1], &mats[2], &mats[3]);
        break;
      case 5:
        mttkrp5_cpu_taco(&mats[0], &B_taco, &mats[1], &mats[2], &mats[3], &mats[4]);
        break;
      default:
        std::cout << "Unsupported number of modes!" << std::endl;
        exit(1);
        break;
    }
    std::cout << "taco vs splatt: " << compare_matrices(Aref_taco, mats[0]) << std::endl;

    switch (B_splatt->nmodes) {
      case 3:
        mttkrp3_cpu_taco_workspace(&mats[0], &B_taco, &mats[1], &mats[2]);
        break;
      case 4:
        mttkrp4_cpu_taco_workspace(&mats[0], &B_taco, &mats[1], &mats[2], &mats[3]);
        break;
      case 5:
        mttkrp5_cpu_taco_workspace(&mats[0], &B_taco, &mats[1], &mats[2], &mats[3], &mats[4]);
        break;
      default:
        std::cout << "Unsupported number of modes!" << std::endl;
        exit(1);
        break;
    }
    std::cout << "taco workspace vs splatt: " << compare_matrices(Aref_taco, mats[0]) << std::endl;

    exit(0);
  }

  const int trials = 25;

  RUN(mttkrp_cpu_splatt(B_splatt, &factor_matrices);,
      trials, "mttkrp", "cpu", "csf", "splatt", tensor_name);
  switch (B_splatt->nmodes) {
    case 3:
      RUN(mttkrp3_cpu_taco_unscheduled(&mats[0], &B_taco, &mats[1], &mats[2]);, 
          trials, "mttkrp", "cpu", "csf", "taco_unscheduled", tensor_name);
      RUN(mttkrp3_cpu_taco(&mats[0], &B_taco, &mats[1], &mats[2]);, 
          trials, "mttkrp", "cpu", "csf", "taco", tensor_name);
      RUN(mttkrp3_cpu_taco_workspace(&mats[0], &B_taco, &mats[1], &mats[2]);, 
          trials, "mttkrp", "cpu", "csf", "taco_workspace", tensor_name);
      break;
    case 4:
      RUN(mttkrp4_cpu_taco_unscheduled(&mats[0], &B_taco, &mats[1], &mats[2], &mats[3]);, 
          trials, "mttkrp", "cpu", "csf", "taco_unscheduled", tensor_name);
      RUN(mttkrp4_cpu_taco(&mats[0], &B_taco, &mats[1], &mats[2], &mats[3]);, 
          trials, "mttkrp", "cpu", "csf", "taco", tensor_name);
      RUN(mttkrp4_cpu_taco_workspace(&mats[0], &B_taco, &mats[1], &mats[2], &mats[3]);, 
          trials, "mttkrp", "cpu", "csf", "taco_workspace", tensor_name);
      break;
    case 5:
      RUN(mttkrp5_cpu_taco_unscheduled(&mats[0], &B_taco, &mats[1], &mats[2], &mats[3], &mats[4]);, 
          trials, "mttkrp", "cpu", "csf", "taco_unscheduled", tensor_name);
      RUN(mttkrp5_cpu_taco(&mats[0], &B_taco, &mats[1], &mats[2], &mats[3], &mats[4]);, 
          trials, "mttkrp", "cpu", "csf", "taco", tensor_name);
      RUN(mttkrp5_cpu_taco_workspace(&mats[0], &B_taco, &mats[1], &mats[2], &mats[3], &mats[4]);, 
          trials, "mttkrp", "cpu", "csf", "taco_workspace", tensor_name);
      break;
    default:
      break;
  }
}

#endif
