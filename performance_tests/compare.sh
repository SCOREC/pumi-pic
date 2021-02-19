#!/bin/bash
# Bash script to run a series of tests of ps_combo at optimal configs

for struct in 0 1 
do 
  ./ps_combo 50000 50000000 1 -n $struct --opimal &> out.txt
  ./ps_combo 50000 50000000 2 -n $struct --optimal &>> out.txt
  ./ps_combo 50000 50000000 3 -n $struct -r 1.0 --optimal &>> out.txt
  python3 combo_format.py 3 1 
done
