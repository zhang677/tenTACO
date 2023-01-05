feat=32
tune=16
threads=512
second=4
for third in 4 8 16 32; do
  python codegen.py --alg eb_pr -s $second -t $third --feat $feat --tb $threads --host --check --call --tune $tune
  ((tune++))
done
second=8
for third in 4 8 16 32; do
  python codegen.py --alg eb_pr -s $second -t $third --feat $feat --tb $threads --host --check --call --tune $tune
  ((tune++))
done
second=16
for third in 4 8 16 32; do
  python codegen.py --alg eb_pr -s $second -t $third --feat $feat --tb $threads --host --check --call --tune $tune
  ((tune++))
done
second=32
for third in 4 8 16 32; do
  python codegen.py --alg eb_pr -s $second -t $third --feat $feat --tb $threads --host --check --call --tune $tune
  ((tune++))
done