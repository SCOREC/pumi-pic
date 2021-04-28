#!/bin/bash
# Bash script to run a series of densely-populated ps_combo tests

cd ~/barn/pumipic_CabM/
source envAimos.sh

# Dense Testing Script for AiMOS: small elm n, large ptcl n
echo "---------------------AiMOS Medium-Sparsity (smallE_largeP)"
for e in 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500
do
  for distribution in 1 2 3
  do 
    for struct in 0 1 2 3
    do
      mpirun -np 2 ./ps_combo160 --kokkos-ndevices=2 $e $((e*10000)) $distribution $struct
    done
  done
done