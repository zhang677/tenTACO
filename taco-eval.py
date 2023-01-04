#!/usr/bin/python3

import os
import sys
import subprocess

matrices_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/nfs_data/zhanggh/mytaco/learn-taco/tensors/'
experiment = sys.argv[2] if len(sys.argv) > 2 else 'spmv_csr_cpu'
results_dir = sys.argv[3] if len(sys.argv) > 3 else '.'
hardware = sys.argv[4] if len(sys.argv) > 4 else 'None'
feature = sys.argv[5] if len(sys.argv) > 5 else '64'



log_path = os.path.join(results_dir, experiment + '.csv')
with open(log_path, 'w') as log_file:
  log_file.write('kernel,platform,format,library,tensor,time\n')

binary = ('./tensor-eval' if experiment.startswith('mttkrp') or experiment.startswith('ttv') else './matrix-eval') + ('-gpu' if experiment.endswith('gpu') else '')
cmd_prefix = []  if experiment.endswith('gpu') else ['numactl', '-N', '0', '-m', '0']

if '_dia_' in experiment or '_ell_' in experiment:
  input_matrices = ['crystm03', 'jnlbrng1', 'obstclae', 'chem_master1', 
                    'dixmaanl', 'shyy161', 'apache1', 'denormal', 'Baumann', 
                    'majorbasis', 'Lin', 'apache2', 'ecology1', 'atmosmodd']
else:
  if matrices_dir == '/home/nfs_data/zhanggh/mytaco/learn-taco/tensors':
    input_matrices = [os.fsencode(input_folder).decode() for input_folder in 
                      sorted(os.listdir(os.fsencode(matrices_dir)))]
    #print(input_matrices)
  else: # matrices_dir == '/home/nfs_data/zhanggh/tenTACO/tensor_subset'
    f = open('part_names.txt','r')
    input_matrices = f.readline().split(',')
    input_matrices.sort()
    f.close()
  
print(input_matrices)
print(len(input_matrices))
for input_matrix in input_matrices:
  input_matrix_dir = os.path.join(matrices_dir, input_matrix)
  cmd = cmd_prefix + [binary, input_matrix_dir, input_matrix, experiment, results_dir, hardware, feature]
  print(' '.join(cmd))
  subprocess.run(cmd)
