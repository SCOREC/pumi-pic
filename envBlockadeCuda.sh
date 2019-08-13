module purge
module load gcc
module load mpich
module load cmake
kk=/lore/cwsmith/develop/build-kokkos-blockade-cuda-opt/install
omega_h=/lore/cwsmith/develop/build-omegah-blockade-cuda-opt/install
export CMAKE_PREFIX_PATH=$kk:$omega_h:$CMAKE_PREFIX_PATH
export MPICH_CXX=/lore/cwsmith/develop/kokkos/bin/nvcc_wrapper
cuda=/usr/local/cuda-10.1
export PATH=$cuda/bin:$PATH
export LD_LIBRARY_PATH=$cuda/lib64:$LD_LIBRARY_PATH
