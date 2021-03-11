#!/bin/bash
# Bash script to run a series of ps_combo tests

# large elm n, large ptcl n
for e in 1000 2000 5000 10000 25000 50000 75000 100000 125000 150000
do
  for distribution in 1 2 3
  do 
    for percent in 50 # 10% and 50% should be very similar
    do 
      for struct in 0 1 2
      do
        mpirun -np 2 ./ps_combo268 $e $((e*1000)) $distribution -p $percent -n $struct
      done
    done
  done
done