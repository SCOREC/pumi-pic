#!/bin/bash
# Bash script to run a series of ps_combo tests

# small elm n, large ptcl n
for e in 250 500 750 1000 1250 1500 1750 2000
do
  for distribution in 1 2 3
  do 
    for percent in 50
    do 
      for struct in 0 1 2
      do
        mpirun -np 2 ./ps_combo268 $e $((e*10000)) $distribution -p $percent -n $struct
      done
    done
  done
done