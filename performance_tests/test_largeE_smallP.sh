#!/bin/bash
# Bash script to run a series of sparsely-populated ps_combo tests

cd ~/barn/pumipic_CabM/
source envAimos.sh

# Medium-Sparsity Testing Script for AiMOS: large elm n, small ptcl n
echo "---------------------AiMOS Medium-Sparsity (largeE_smallP)"
for e in 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000
do
  for distribution in 1 2 3
  do 
    for struct in 0 1 2 3
    do
      mpirun -np 2 ./ps_combo160 --kokkos-ndevices=2 $e $((e*1000)) $distribution $struct
    done
  done
done