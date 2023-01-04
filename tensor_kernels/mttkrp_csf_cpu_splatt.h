#ifndef MTTKRP_CPU_SPLATT_H
#define MTTKRP_CPU_SPLATT_H

#include "ds.h"
#include "timers.h"

#include "splatt.h"

void mttkrp_cpu_splatt(splatt_csf *B, splatt_kruskal *mats) { 
  double *cpd_opts = splatt_default_opts();
  cpd_opts[SPLATT_OPTION_NTHREADS] = omp_get_num_threads();
  cpd_opts[SPLATT_OPTION_NITER] = 1;
  cpd_opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  cpd_opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;

  const int mode = B->dim_perm[0];
TIME_COLD(
  splatt_mttkrp(mode, mats->rank, B, mats->factors, mats->factors[mode], cpd_opts);
)
}

#endif
