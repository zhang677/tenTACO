#include "ds.h"
#include "timers.h"
#include "tensor_experiments/mttkrp_csf_gpu.h"
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>

#include "splatt.h"

Timer timer;
std::ofstream log_file;

int main(int argc, char* argv[]) {
  const std::string tensor_dir  = (argc > 1) ? argv[1] : "/home/nfs_data/zhanggh/mytaco/learn-taco/tensors/";
  const std::string tensor_name = (argc > 2) ? argv[2] : "nell-2";
  const std::string experiment  = (argc > 3) ? argv[3] : "mttkrp_csf_gpu";
  const std::string results_dir = (argc > 4) ? argv[4] : ".";
  const std::string hardware       = (argc > 5) ? argv[5] : "3090";
  const std::string feature_dim    = (argc > 6) ? argv[6] : "32";


  const bool do_verify = true;
  const int num_cols = std::stoi(feature_dim);
  const std::string tensor_path = tensor_dir + "/" + tensor_name + ".tns";
  splatt_csf* input_tensor = read_tensor(tensor_path);

  const std::string log_path = results_dir + "/" + experiment + ".csv";
  log_file.open(log_path, std::ofstream::app);
 
  if (experiment == "mttkrp_csf_gpu") {
    mttkrp_csf_gpu(input_tensor, tensor_name, do_verify, num_cols);
  }
 else if (experiment == "ttv_csf_gpu") {
    // ttv_csf_gpu(input_tensor, tensor_name, do_verify);
  }

  log_file.close();

  return 0;
}
