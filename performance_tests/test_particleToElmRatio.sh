#!/bin/bash
[[ $# != 3 ]] && echo "usage: $0 <binDir> <particles per element> <elm=large|small>" && exit 1

binDir=$1
[[ ! -d "$binDir" ]] && echo "binary dir $binDir does not exist" && exit 1

ptclsPerElm=$2
[[ "$ptclsPerElm" -le 0 ]] && echo "particles per element must be greater than zero" && exit 1

elms=$3
[[ "$elms" != "large" && "$elms" != "small" ]] && echo "elms must be \"large\" or \"small\"" && exit 1

large="10000 15000 20000 25000 30000 35000 40000 45000 50000 55000"
small="1000 1500 2000 2500 3000 3500 4000 4500 5000 5500"
elmRange=$large
[[ "$elms" == "small" ]] && elmRange=$small

echo "--------------------- Test: ${elms}E_${ptclsPerElm}P"
for e in $elmRange
do
  for distribution in 1 2 3
  do 
    for struct in 0 1 2 3
    do
      cmd="srun -n 2 $binDir/ps_combo160 --kokkos-map-device-id-by=mpi_rank $e $((e*ptclsPerElm)) $distribution $struct"
      if [[  "$TEST_DRY_RUN" ]]; then
        echo "$cmd"
      else
        $cmd
      fi 
    done
  done
done
