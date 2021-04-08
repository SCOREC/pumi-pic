#!/bin/bash
# Bash script to run a series of ps_combo tests

# Medium-Sparsity Testing Script for AiMOS: large elm n, small ptcl n
for e in 10000 20000 30000 40000 50000 60000 70000
do
  for distribution in 1 2 3
  do 
    for struct in 0 1 2
    do
      mpirun -np 2 ./ps_combo160 --kokkos-ndevices=2 $e $((e*1000)) $distribution $struct
    done
  done
done