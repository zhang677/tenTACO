feat=32
threads=256
tune=0
for second in 4 8 16 32; do
  for third in 1 2 4 8; do
    python codegen.py --alg eb_sr -s $second -t $third --feat $feat --tb $threads --host --check --call --tune $tune
    ((tune++))
  done
done
threads=512
for second in 4 8 16 32; do
  for third in 1 2 4 8; do
    python codegen.py --alg eb_sr -s $second -t $third --feat $feat --tb $threads --host --check --call --tune $tune
    ((tune++))
  done
done