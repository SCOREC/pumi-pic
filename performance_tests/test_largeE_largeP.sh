#!/bin/bash
# Bash script to run a series of ps_combo tests

# large elm n, large ptcl n
for e in 1000 2000 5000 10000 25000 50000 75000 100000 125000 150000
do
  for distribution in 1 2 3
  do 
    for percent in 10 50
    do 
      for struct in 0 1 2
      do
        ./ps_combo $e $((e*1000)) $distribution -p $percent -n $struct # Blockade
        #mpirun -np 1 ./ps_combo $e $((e*1000)) $distribution -p $percent -n $struct # AiMOS
      done
    done
  done
done