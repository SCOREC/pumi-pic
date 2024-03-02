#!/bin/bash

runDir=$PWD
binDir=/path/to/pumipic/build/performance_tests

function runTest() {
  ptclsPerElm=$1
  elms=$2
  tname=${elms}E_${ptclsPerElm}P
  ./test_particleToElmRatio.sh $binDir $ptclsPerElm $elms off &> ${tname}_$SLURM_JOB_ID.txt
  ./test_particleToElmRatio.sh $binDir $ptclsPerElm $elms on &> ${tname}_Optimal_$SLURM_JOB_ID.txt
  echo "test_${tname} DONE"
}

runTest 1000  large
runTest 100   large
runTest 10    large
runTest 5     large
