#!/bin/bash -x

#load system modules
source /etc/profile.d/modules.sh
source /etc/profile

module load gcc/7.3.0-bt47fwr mpich/3.2.1-niuhmad cmake/3.12.1-wfk2b7e
export CMAKE_PREFIX_PATH=/lore/cwsmith/develop/build-cabana-cuda-blockade/install/:$CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=/lore/cwsmith/develop/trilinos/install-kokkos-openmp-cuda-blockade:$CMAKE_PREFIX_PATH
export MPICH_CXX=/lore/cwsmith/develop/trilinos/src/packages/kokkos/bin/nvcc_wrapper
cuda=/usr/local/cuda-9.2
export PATH=$cuda/bin:$PATH
export LD_LIBRARY_PATH=$cuda/lib64:$LD_LIBRARY_PATH

d=/fasttmp/cwsmith/nightlyBuilds
cd $d/repos/particle_structures
git pull
cd $d
#remove old compilation
[ -d build_particle_structures ] && rm -rf build_particle_structures/

#run nightly test script
ctest -VV -D Nightly -S $d/repos/particle_structures/cdash/nightly.cmake
