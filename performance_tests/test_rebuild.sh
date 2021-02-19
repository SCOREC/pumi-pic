#!/bin/bash
# Bash script to run a series of tests of ps_rebuild

for e in 10 25 50 100 250 
do
  for distribution in 0 1 2 
  do
    for percent in 50 75 90
    do
      ./ps_rebuild $e $((e * 100000)) $distribution $percent
    done
  done
done
