#!/bin/bash

name=pumi-pic

cd $SCRATCH/globus-compute/$name-test

export root=$PWD
module load cmake
export MPICH_CXX=$root/kokkos/bin/nvcc_wrapper

cd build-$name
salloc --time 00:20:00 --constrain=gpu --qos=interactive --nodes=1 --ntasks-per-node=40 --cpus-per-task=1 --gpus=1 --account=m4564 ctest
cat $PWD/Testing/Temporary/LastTest.log