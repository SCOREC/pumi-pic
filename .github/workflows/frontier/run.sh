#!/bin/bash

name=pumi-pic

cd /lustre/orion/phy122/scratch/castia5/globus-compute/$name-test
source env.sh

cd build-$name
salloc --account=PHY122 --time=00:20:00 -q debug --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=1 --gpus=1 ctest
cat $PWD/Testing/Temporary/LastTest.log