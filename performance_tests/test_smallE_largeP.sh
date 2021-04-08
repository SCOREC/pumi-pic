#!/bin/bash
# Bash script to run a series of ps_combo tests

# Dense Testing Script for AiMOS: small elm n, large ptcl n
for e in 1000 2000 3000 4000 5000 6000 7000
do
  for distribution in 1 2 3
  do 
    for struct in 0 1 2
    do
      mpirun -np 2 ./ps_combo160 --kokkos-ndevices=2 $e $((e*10000)) $distribution $struct
    done
  done
done