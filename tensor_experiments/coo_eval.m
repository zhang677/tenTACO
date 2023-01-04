function coo_eval(tensor_dir, tensor_name, experiment, results_dir, validate)
  addpath('/data/scratch/s3chou/tensor_toolbox-v3.1/');
  
  mex COMPFLAGS='$COMPFLAGS -O3 -march=native -mtune=native -ffast-math' ../tensor_kernels/fmtaco.cpp
  
  if ~exist('tensor_dir', 'var')
    tensor_dir = '/data/scratch/s3chou/frostt/vast-2015-mc1-3d/';
  end
  if ~exist('tensor_name', 'var')
    tensor_name = 'vast-2015-mc1-3d';
  end
  if ~exist('experiment', 'var')
    experiment = 'ttv_coo_cpu';
  end
  if ~exist('results_dir', 'var')
    results_dir = '.';
  end
  if ~exist('validate', 'var')
    validate = 0;
  end
  
  num_trials = 25;
  dummyarr = rand(1,30000000);
  dummyidx = randperm(length(dummyarr));
  
  tensor_path = strcat(tensor_dir, '/', tensor_name, '.tns');
  L = dlmread(tensor_path);
  %[~, didx] = sort(max(L(:,1:3)));
  X = sptensor(L(:,1:3), L(:,4));
  dims = X.size;
  clear L;
  
  Xz = int32(X.subs) - 1;
  if strcmp(experiment, 'mttkrp_coo_cpu') || strcmp(experiment, 'tenadd_coo_cpu') || strcmp(experiment, 'ttv_coo_cpu')
    Xi = sptensor(int32(X.subs), X.vals);
    if validate == 0
      clear X;
    end
  end
  
  log_path = strcat(results_dir, '/', experiment, '.csv');
  log_file = fopen(log_path, 'at');
  
  if strcmp(experiment, 'mttkrp_coo_cpu')
    F = 32;
    M1 = rand(dims(1), F);
    M2 = rand(dims(2), F);
    M3 = rand(dims(3), F);
  
    if validate
      T = fmtaco(Xz, Xi.vals, int32(Xi.size), [], M2', int32(size(M2)), [], M3', int32(size(M3)), 'mttkrp')';
      V = mttkrp(Xi, {M1, M2, M3}, 1);
      fprintf('mttkrp_coo_cpu max diff: %d\n', max(max(abs(T - V))));
      clear T;
      clear V;
    end
  
    for i = 1:num_trials
      dummysum = sum(dummyarr(dummyidx));
      tic;
      T = fmtaco(Xz, Xi.vals, int32(Xi.size), [], M2', int32(size(M2)), [], M3', int32(size(M3)), 'mttkrp')';
      time = toc;
      clear T;
      log_result(log_file, 'mttkrp', 'taco', tensor_name, i, time);
    end
    for i = 1:num_trials
      dummysum = sum(dummyarr(dummyidx));
      tic;
      T = mttkrp(Xi, {M1, M2, M3}, 1);
      time = toc;
      clear T;
      log_result(log_file, 'mttkrp', 'ttb', tensor_name, i, time);
    end
  elseif strcmp(experiment, 'ttv_coo_cpu')
    v = rand(dims(3), 1);
  
    if validate
      T = ttv(Xi, v, 3);
      [Bval, Bind] = fmtaco(Xz, Xi.vals, int32(Xi.size), [], v, int32(size(v)), 'ttv');
      V = sptensor(double(Bind + 1), Bval);
      fprintf('ttv_coo_cpu validated: %d\n', isequal(T, V));
      clear T;
      clear V;
      clear Bval;
      clear Bind;
    end
  
    for i = 1:num_trials
      dummysum = sum(dummyarr(dummyidx));
      tic;
      [Bval, Bind] = fmtaco(Xz, Xi.vals, int32(Xi.size), [], v, int32(size(v)), 'ttv');
      time = toc;
      clear Bval;
      clear Bind;
      log_result(log_file, 'ttv', 'taco', tensor_name, i, time);
    end
    for i = 1:num_trials
      dummysum = sum(dummyarr(dummyidx));
      tic;
      T = ttv(Xi, v, 3);
      time = toc;
      clear T;
      log_result(log_file, 'ttv', 'ttb', tensor_name, i, time);
    end
  elseif strcmp(experiment, 'ttm_coo_cpu')
    F = 16;
    N3 = rand(F, dims(3));
    not_too_large = ((dims(1) * dims(2) * F * 8) < 2^36);
  
    if validate
      if not_too_large
        T = ttm(X, N3, 3);
        [Cval, Cind] = fmtaco(Xz, X.vals, int32(X.size), [], N3', int32(size(N3)), 'ttm');
        V = sptensor(double(Cind + 1), Cval);
        fprintf('ttm_coo_cpu validated: %d\n', isequal(T, V));
        clear T;
        clear V;
        clear Cval;
        clear Cind;
      else
        fprintf('skip ttm_coo_cpu validation\n');
      end
    end
  
    for i = 1:num_trials
      dummysum = sum(dummyarr(dummyidx));
      tic;
      [Cval, Cind] = fmtaco(Xz, X.vals, int32(X.size), [], N3', int32(size(N3)), 'ttm');
      time = toc;
      clear Cval;
      clear Cind;
      log_result(log_file, 'ttm', 'taco', tensor_name, i, time);
    end
    if not_too_large
      for i = 1:num_trials
        dummysum = sum(dummyarr(dummyidx));
        tic;
        T = ttm(X, N3, 3);
        time = toc;
        clear T;
        log_result(log_file, 'ttm', 'ttb', tensor_name, i, time);
      end
    end
  elseif strcmp(experiment, 'tenadd_coo_cpu')
    idx = Xi.subs;
    idx(:,2) = mod(idx(:,2), Xi.size(2)) + 1;
    Yi = sptensor(idx, Xi.vals, dims);
    clear idx;
    Yz = Yi.subs - 1;
  
    if validate
      T = plus(Xi, Yi);
      [Bval, Bind] = fmtaco(Xz, Xi.vals, int32(Xi.size), Yz, Yi.vals, int32(Yi.size), 'plus');
      V = sptensor(double(Bind + 1), Bval);
      fprintf('tenadd_coo_cpu validated: %d\n', isequal(T, V));
      clear T;
      clear V;
      clear Bval;
      clear Bind;
    end
  
    for i = 1:num_trials
      dummysum = sum(dummyarr(dummyidx));
      tic;
      [Bval, Bind] = fmtaco(Xz, Xi.vals, int32(Xi.size), Yz, Yi.vals, int32(Yi.size), 'plus');
      time = toc;
      clear Bval;
      clear Bind;
      log_result(log_file, 'tenadd', 'taco', tensor_name, i, time);
    end
    for i = 1:num_trials
      dummysum = sum(dummyarr(dummyidx));
      tic;
      T = plus(Xi, Yi);
      time = toc;
      clear T;
      log_result(log_file, 'tenadd', 'ttb', tensor_name, i, time);
    end
  end
  
  fclose(log_file)
end

function log_result(log_file, kernel, library, tensor, trial, time)
  timestamp = int64(posixtime(datetime('now'))) * 10^9;
  trial = int32(trial);
  time = time * 10^3;
  fprintf(log_file, '%s,cpu,coo,%s,%s,%d,unknown_hash,%d,%f\n', kernel, library, tensor, trial, timestamp, time);
end
