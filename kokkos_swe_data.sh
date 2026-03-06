#!/bin/bash

# generate data for shallow water equation
THREADS=20
INTEROP_THREADS=1
export OMP_NUM_THREADS=${THREADS}
export OMP_INTEROP_THREADS=${INTEROP_THREADS}

N_CPU=(500 1000 2000 3000 4000 5000)
N_GPU=(500 750 1000 1250 1500 1750 2000 2250 2500)

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