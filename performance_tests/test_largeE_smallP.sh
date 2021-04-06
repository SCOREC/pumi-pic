#!/bin/bash
# Bash script to run a series of ps_combo tests

# Medium-Sparsity Testing Script for AiMOS: large elm n, small ptcl n
for e in 100 200 500 1000 2000 5000 10000 20000
do
  for distribution in 1 2 3 # Even Distribution currently BAD on CabM
  do 
    for struct in 0 1 2
    do
      mpirun -np 2 ./ps_combo160 $e $((e*1000)) $distribution $struct
    done
  done
done