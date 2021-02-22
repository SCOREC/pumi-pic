#!/bin/bash
# Bash script to run a series of ps_combo tests

for e in 0 250 500 750 1000 1250 1500 1750 2000
do
  for distribution in 0 1 2 3
  do 
    for percent in 50
    do 
      for struct in 0 1 2
      do
        ./ps_combo $e $((e*1000)) $distribution -p $percent -n $struct
      done
    done
  done
done
