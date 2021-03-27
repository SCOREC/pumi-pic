#!/bin/bash
# Bash script to run a series of ps_combo tests

# Small Testing Script for Blockade: small elm n, small ptcl n
for e in 250 500 750 1000 1250 1500 1750 2000
do
  for distribution in 0 1 2 3 # Even Distribution currently BAD on CabM
  do 
    for percent in 0.5
    do 
      for struct in 0 1 2
      do
        ./ps_combo160 $e $((e*1000)) $distribution -p $percent -n $struct
      done
    done
  done
done
