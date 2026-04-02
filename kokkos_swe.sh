#! /bin/bash

export OMP_NUM_THREADS=32
export OMP_PLACES="{0:32}"
export OMP_PROC_BIND=close
export OMP_DISPLAY_ENV=verbose
export OMP_DISPLAY_AFFINITY=TRUE

N_CPU=(500 1000 1500 2000 2500 3000)
# N_GPU=(500 1000 1500 2000 2500 3000)

for n in ${N_CPU[@]}
do
    cd ${HOME}/kokkos
    python3 example/shallow_water_equation/swe_export_to_binary.py \
        ~/.easier/triangular_${n}.hdf5 \
        ~/.easier/SW_${n}.hdf5 \
        --output-dir data/swe_binary
    cd ${HOME}/kokkos/example/shallow_water_equation

    delta_t=$(echo "scale=8; 0.5/$n" | bc)
    # openmp
    ./build-omp-release/shallow_water_pipeline \
        --data ${HOME}/kokkos/data/swe_binary \
        --profile --dt ${delta_t} --profile-warmup 5 --profile-iters 20 \
        --output swe_profile_cpu_simple

    # openmp fused
    ./build-omp-release/shallow_water_pipeline_fused \
        --data ${HOME}/kokkos/data/swe_binary \
        --profile --dt ${delta_t} --profile-warmup 5 --profile-iters 20 \
        --output swe_profile_cpu_fused
done


for n in ${N_GPU[@]}
do
    cd ${HOME}/kokkos
    python3 example/shallow_water_equation/swe_export_to_binary.py \
        ~/.easier/triangular_${n}.hdf5 \
        ~/.easier/SW_${n}.hdf5 \
        --output-dir data/swe_binary
    cd ${HOME}/kokkos/example/shallow_water_equation

    delta_t=$(echo "scale=8; 0.5/$n" | bc)
    # cuda
    ./build-volta70/shallow_water_pipeline \
        --data ${HOME}/kokkos/data/swe_binary \
        --profile --dt ${delta_t} --profile-warmup 50 --profile-iters 100 \
        --output swe_profile_cuda
    
    ./build-volta70/shallow_water_pipeline_fused \
        --data ${HOME}/kokkos/data/swe_binary \
        --profile --dt ${delta_t} --profile-warmup 50 --profile-iters 100 \
        --output swe_profile_cuda_fused
done
