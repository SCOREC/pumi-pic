#!/bin/bash

runDir=$PWD
binDir=/path/to/pumipic/build/performance_tests

function runTest() {
  ptclsPerElm=$1
  elms=$2
  tname=${elms}E_${ptclsPerElm}P
  ./test_particleToElmRatio.sh $binDir $ptclsPerElm $elms &> ${tname}_$SLURM_JOB_ID.txt
  cd $runDir
  python output_convert.py ${tname}_$SLURM_JOB_ID.txt ${tname}_rebuild.dat ${tname}_push.dat ${tname}_migrate.dat
  echo "test_${tname} DONE"
}

#runTest 10000 large #OOM
runTest 1000  large
runTest 100   large
runTest 10    large
runTest 5     large
