#! /bin/bash
HOEM_DIR=/mnt/local-fast/qpan

export OMP_NUM_THREADS=40
export OMP_PLACES="{0:40}"
export OMP_PROC_BIND=close
# export OMP_DISPLAY_ENV=verbose
# export OMP_DISPLAY_AFFINITY=TRUE

export LD_PRELOAD=$CONDA_PREFIX/lib/libtcmalloc.so.4:$LD_PRELOAD

# N_CPU=(1000 2000 3000 4000)
N_GPU=(1000 2000 3000 4000)

for n in ${N_CPU[@]}
do
    cd ${HOEM_DIR}/kokkos
    python3 example/shallow_water_equation/swe_export_to_binary.py \
        ${HOEM_DIR}/.easier/triangular_${n}.hdf5 \
        ${HOEM_DIR}/.easier/SW_${n}.hdf5 \
        --output-dir data/swe_binary
    cd ${HOEM_DIR}/kokkos/example/shallow_water_equation

    delta_t=$(echo "scale=8; 0.5/$n" | bc)
    # openmp
    ./build-omp-release/shallow_water_pipeline \
        --data ${HOEM_DIR}/kokkos/data/swe_binary \
        --profile --dt ${delta_t} --profile-warmup 5 --profile-iters 20 \
        --output swe_profile_cpu_simple

    # openmp fused
    ./build-omp-release/shallow_water_pipeline_fused \
        --data ${HOEM_DIR}/kokkos/data/swe_binary \
        --profile --dt ${delta_t} --profile-warmup 5 --profile-iters 20 \
        --output swe_profile_cpu_fused
done


for n in ${N_GPU[@]}
do
    cd ${HOEM_DIR}/kokkos
    python3 example/shallow_water_equation/swe_export_to_binary.py \
        ${HOEM_DIR}/.easier/triangular_${n}.hdf5 \
        ${HOEM_DIR}/.easier/SW_${n}.hdf5 \
        --output-dir data/swe_binary
    cd ${HOEM_DIR}/kokkos/example/shallow_water_equation

    delta_t=$(echo "scale=8; 0.5/$n" | bc)
    # cuda
    ./build-hopper90/shallow_water_pipeline \
        --data ${HOEM_DIR}/kokkos/data/swe_binary \
        --profile --dt ${delta_t} --profile-warmup 50 --profile-iters 100 \
        --output swe_profile_cuda
    
    ./build-hopper90/shallow_water_pipeline_fused \
        --data ${HOEM_DIR}/kokkos/data/swe_binary \
        --profile --dt ${delta_t} --profile-warmup 50 --profile-iters 100 \
        --output swe_profile_cuda_fused
done

HOEM_DIR=/mnt/local-fast/qpan

cd ${HOEM_DIR}/kokkos
python3 example/shallow_water_equation/swe_export_to_binary.py \
    ${HOEM_DIR}/.easier/triangular_2000.hdf5 \
    ${HOEM_DIR}/.easier/SW_2000.hdf5 \
    --output-dir data/swe_binary
cd ${HOEM_DIR}/kokkos/example/shallow_water_equation


export OMP_NUM_THREADS=40 & nsys profile -o kokkos  --stats=true  --trace=cuda,osrt,nvtx,openmp,mpi,cublas    --cudabacktrace=all  --force-overwrite=true --delay 5 \
    ./build-hopper90/shallow_water_pipeline_fused \
        --data /mnt/local-fast/qpan/kokkos/data/swe_binary \
        --profile --dt 0.0005 --profile-warmup 50 --profile-iters 100 \
        --output swe_profile_cuda_fused