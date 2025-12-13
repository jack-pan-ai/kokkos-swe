#! /bin/bash

export EASIER_DISABLE_JIT_HASH=1

THREADS=20
INTEROP_THREADS=1
export OMP_NUM_THREADS=${THREADS}
export OMP_INTEROP_THREADS=${INTEROP_THREADS}

N_GPU=(500 1000 2000 3000 4000 5000)
N_CPU=(500 750 1000 1250 1500 1750 2000 2250 2500)

for n in ${N_CPU[@]}
do
    cd /home/panq/dev/FlexSpmv/kokkos
    python3 example/shallow_water_equation/swe_export_to_binary.py \
        ~/.easier/triangular_${n}.hdf5 \
        ~/.easier/SW_${n}.hdf5 \
        --output-dir data/swe_binary
    cd /home/panq/dev/FlexSpmv/kokkos/example/shallow_water_equation

    delta_t=$(echo "scale=8; 0.5/$n" | bc)
    # openmp
    ./build-omp/shallow_water_pipeline \
        --data /home/panq/dev/FlexSpmv/kokkos/data/swe_binary \
        --profile --dt ${delta_t} --profile-warmup 5 --profile-iters 50 \
        --output swe_profile_cpu
done


for n in ${N_GPU[@]}
do
    cd /home/panq/dev/FlexSpmv/kokkos
    python3 example/shallow_water_equation/swe_export_to_binary.py \
        ~/.easier/triangular_${n}.hdf5 \
        ~/.easier/SW_${n}.hdf5 \
        --output-dir data/swe_binary
    cd /home/panq/dev/FlexSpmv/kokkos/example/shallow_water_equation

    delta_t=$(echo "scale=8; 0.5/$n" | bc)
    # cuda
    ./build-volta70/shallow_water_pipeline \
        --data /home/panq/dev/FlexSpmv/kokkos/data/swe_binary \
        --profile --dt ${delta_t} --profile-warmup 20 --profile-iters 100 \
        --output swe_profile_cuda
done
