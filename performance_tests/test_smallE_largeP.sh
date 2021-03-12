#!/bin/bash
# Bash script to run a series of ps_combo tests

# Dense Testing Script for AiMOS: small elm n, large ptcl n
for e in 100 200 500 1000 1200 1500 2000 5000 10000
do
  for distribution in 1 2 3 # Even Distribution currently BAD on CabM
  do 
    for percent in 50 # 10% and 50% should be very similar
    do 
      for struct in 0 1 2
      do
        mpirun -np 2 ./ps_combo160 $e $((e*10000)) $distribution -p $percent -n $struct
      done
    done
  done
done