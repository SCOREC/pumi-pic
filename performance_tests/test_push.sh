#!/bin/bash
# Bash script to run a series of tests of ps_pseudopush

for e in 25 
do
  for distribution in 0 3  
  do
    ./ps_pseudopush $e $((e * 10000)) $distribution 0
  done
done
