#ifndef TTV_CSF_GPU_H
#define TTV_CSF_GPU_H

#include "ds.h"
#include "timers.h"

#include "gpu_kernels.cuh"

void ttv_csf_cpu_taco_ref(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *c) {
TIME_COLD(
  float* restrict A_vals = (float*)(A->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int* restrict B2_pos = (int*)(B->indices[1][0]);
  int* restrict B2_crd = (int*)(B->indices[1][1]);
  int* restrict B3_pos = (int*)(B->indices[2][0]);
  int* restrict B3_crd = (int*)(B->indices[2][1]);
  float* restrict B_vals = (float*)(B->vals);
  int c1_dimension = (int)(c->dimensions[0]);
  float* restrict c_vals = (float*)(c->vals);
  int a1_dimension = (int)(A->dimensions[0]);

  int jB = 0;

    _Pragma("omp parallel for schedule(static)")
  for (int py = 0; py < a1_dimension; py++) {
    A_vals[py] = 0.0;
  }

  _Pragma("omp parallel for")
  for (int i = 0; i < B1_dimension; i++) {
    for (int jB = B2_pos[i]; jB < B2_pos[(i + 1)]; jB++) {
      for (int kB = B3_pos[jB]; kB < B3_pos[(jB + 1)]; kB++) {
        int k = B3_crd[kB];
        A_vals[jB] = A_vals[jB] + B_vals[kB] * c_vals[k];
      }
    }
  }
  );
}

void ttv_csf_gpu(splatt_csf* B_splatt, const std::string& tensor_name, const bool do_verify) {
  taco_tensor_t B_taco = to_taco_tensor(B_splatt);

  const int J = 32;

  int B1_dimension = (int)(B_taco.dimensions[0]);
  int B3_dimension = (int)(B_taco.dimensions[2]);
  int* restrict B2_pos = (int*)(B_taco.indices[1][0]);

  long out_size = B2_pos[B1_dimension];

    EigenVector x_eigen0 = gen_vector(B3_dimension);
  taco_tensor_t x_taco0 = to_taco_tensor(x_eigen0);


    EigenVector x_eigen = gen_vector(B3_dimension);
  taco_tensor_t x_taco = to_taco_tensor(x_eigen);

   double* restrict x_vals0 = (double*)(x_taco0.vals);
   float* restrict x_vals = (float*)(x_taco.vals);
   for(long i=0; i<B3_dimension; i++) {
	x_vals[i] = (float)x_vals0[i];
   }

  EigenVector y_eigen = gen_vector(out_size);
  taco_tensor_t y_taco = to_taco_tensor(y_eigen);

  if (do_verify) {
	
	  EigenVector y_eigen_ref = gen_vector(out_size);
	  taco_tensor_t y_taco_ref = to_taco_tensor(y_eigen_ref);

	  ttv_csf_cpu_taco_ref(&y_taco_ref, &B_taco, &x_taco);
	  ttv_csf_gpu_taco(&y_taco, &B_taco, &x_taco);

	  std::cout << "taco_unscheduled vs taco: " << compare_vectors_float(y_taco_ref, y_taco) << std::endl;
	exit(0);
	
  }

  const int trials = 25;
     RUN_GPU(ttv_csf_gpu_taco(&y_taco, &B_taco, &x_taco);,
          trials, "mttkrp", "gpu", "csf", "taco", tensor_name);


}

#endif
