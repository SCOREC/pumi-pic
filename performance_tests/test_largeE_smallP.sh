#!/bin/bash
# Bash script to run a series of ps_combo tests

# large elm n, small ptcl n
for e in 2500 5000 7500 10000 12500 15000 17500 20000
do
  for distribution in 1 2 3
  do 
    for percent in 50
    do 
      for struct in 0 1 2
      do
        mpirun -np 1 ./ps_combo36 $e $((e*1000)) $distribution -p $percent -n $struct
        #mpirun -np 1 ./ps_combo268 $e $((e*1000)) $distribution -p $percent -n $struct
        #mpirun -np 1 ./ps_migrate268 $e $((e*1000)) $distribution -p $percent -n $struct #remove CSR to run
      done
    done
  done
done