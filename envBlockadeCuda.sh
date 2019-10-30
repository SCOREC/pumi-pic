module purge
module load gcc
module load mpich
module load cmake
module load netcdf
module load vim
ncxx=/lore/gopan/install/build-netcdfcxx431/install
export NETCDF_PREFIX=$ncxx

kk=/lore/cwsmith/develop/build-kokkos-blockade-cuda/install
omega_h=/lore/cwsmith/develop/build-omegah-blockade-cuda/install
export CMAKE_PREFIX_PATH=$kk:$omega_h:$ncxx:$CMAKE_PREFIX_PATH
export MPICH_CXX=/lore/cwsmith/develop/kokkos/bin/nvcc_wrapper
cuda=/usr/local/cuda-10.1
export PATH=$cuda/bin:$PATH
export LD_LIBRARY_PATH=$cuda/lib64:$LD_LIBRARY_PATH

# /lore/gopan/install/build-blockade-ps-noOpt/install
