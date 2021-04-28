#!/bin/bash
# Bash script to run a series of ps_combo tests

# Dense Testing Script for AiMOS: small elm n, large ptcl n
for e in 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500
do
  for distribution in 1 2 3
  do 
    for struct in 0 1 2
    do
      mpirun -np 2 ./ps_combo160 --kokkos-ndevices=2 $e $((e*10000)) $distribution $struct
    done
  done
done