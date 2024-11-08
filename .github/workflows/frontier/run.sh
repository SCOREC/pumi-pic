#!/bin/bash

name=pumi-pic

cd /lustre/orion/phy122/scratch/castia5/globus-compute/$name-test

module load rocm
module load craype-accel-amd-gfx90a
module load cray-mpich
export CRAYPE_LINK_TYPE=dynamic
export MPICH_GPU_SUPPORT_ENABLED=1

cd build-$name
salloc --account=PHY122 --time=00:20:00 -q debug --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=1 --gpus=1 ctest
cat $PWD/Testing/Temporary/LastTest.log