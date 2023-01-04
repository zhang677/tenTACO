#ifndef TTV_CSF_CPU_H
#define TTV_CSF_CPU_H

#include "ds.h"
#include "timers.h"

#include "tensor_kernels/ttv_csf_cpu_taco_unscheduled.h"
#include "tensor_kernels/ttv_csf_cpu_taco.h"

void ttv_csf_cpu(splatt_csf* B_splatt, const std::string& tensor_name, const bool do_verify) {
  taco_tensor_t B_taco = to_taco_tensor(B_splatt);

  if( B_splatt->nmodes != 3) exit(0);

  const int J = 32;

  int B1_dimension = (int)(B_taco.dimensions[0]);
  int B3_dimension = (int)(B_taco.dimensions[2]);
  long* restrict B2_pos = (long*)(B_taco.indices[1][0]);

  long out_size = B2_pos[B1_dimension];

    EigenVector x_eigen = gen_vector(B3_dimension);
  taco_tensor_t x_taco = to_taco_tensor(x_eigen);

  EigenVector y_eigen = gen_vector(out_size);
  taco_tensor_t y_taco = to_taco_tensor(y_eigen);

  if (do_verify) {
	  EigenVector y_eigen_ref = gen_vector(out_size);
	  taco_tensor_t y_taco_ref = to_taco_tensor(y_eigen_ref);

	  ttv_csf_cpu_taco_unscheduled(&y_taco_ref, &B_taco, &x_taco);
	  ttv_csf_cpu_taco(&y_taco, &B_taco, &x_taco);

	  std::cout << "taco_unscheduled vs taco: " << compare_vectors(y_taco_ref, y_taco) << std::endl;
	exit(0);
  }

  const int trials = 25;
     RUN(ttv_csf_cpu_taco_unscheduled(&y_taco, &B_taco, &x_taco);,
          trials, "ttv", "cpu", "csf", "taco_unscheduled", tensor_name);
     RUN(ttv_csf_cpu_taco(&y_taco, &B_taco, &x_taco);,
          trials, "ttv", "cpu", "csf", "taco", tensor_name);


}

#endif
