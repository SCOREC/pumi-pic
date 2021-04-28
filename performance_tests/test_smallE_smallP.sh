#!/bin/bash
# Bash script to run a series of ps_combo tests

# Small Testing Script for Blockade: small elm n, small ptcl n
for e in 250 500 750 1000 1250 1500 1750 2000
do
  for distribution in 0 1 2 3
  do 
    for struct in 0 1 2 3
    do
      ./build-pumipic-blockade-cuda/performance_tests/ps_combo160 --kokkos-ndevices=2 $e $((e*1000)) $distribution $struct
    done
  done
done
