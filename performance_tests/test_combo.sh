#!/bin/bash
# Bash script to run a series of ps_combo tests

for e in 1000 5000 #10000 25000 38000 50000 75000 100000
do
  for distribution in 3
  do 
    for percent in 10
    do 
      for struct in 0 1 2
      do
        ./ps_combo $e $((e*1000)) $distribution -n $struct
      done
    done
  done
done
