#!/bin/bash

export OMP_NUM_THREADS=20
export OMP_PLACES="{0:20}"
export OMP_PROC_BIND=close
export OMP_DISPLAY_ENV=verbose
export OMP_DISPLAY_AFFINITY=TRUE

export LD_PRELOAD=$CONDA_PREFIX/lib/libtcmalloc.so.4:$LD_PRELOAD

N_CPU=(500 1000 1500 2000 2500 3000)
N_GPU=(500 1000 1500 2000 2500 3000)

for n in ${N_CPU[@]}
do
python3 example/shallow_water_equation/swe_export_to_binary.py \
    ~/.easier/triangular_${n}.hdf5 \
    ~/.easier/SW_${n}.hdf5 \
    --output-dir data/swe_binary
done

for n in ${N_GPU[@]}
do
python3 example/shallow_water_equation/swe_export_to_binary.py \
    ~/.easier/triangular_${n}.hdf5 \
    ~/.easier/SW_${n}.hdf5 \
    --output-dir data/swe_binary
done