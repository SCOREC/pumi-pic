module load gcc
module load mpich
module load cmake
module load netcdf
ncxx=/lore/gopan/install/build-netcdfcxx431/install
#export NetCDF_PREFIX=$ncxx
export MY_NCXX=$ncxx #/lib64/cmake/netCDFCxx
kk=/lore/gopan/install/build-kokkos-cuda-debug-profile-rhel7/install
omega_h=/lore/gopan/install/build-omegah-cuda-rhel7/install
cuda=/usr/local/cuda-10.1
export PATH=$cuda/bin:$PATH
export LD_LIBRARY_PATH=$cuda/lib64:$ncxx/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${kk}:$omega_h:$ncxx:$CMAKE_PREFIX_PATH
kksrc=/lore/gopan/install/kokkos
export MPICH_CXX=${kksrc}/bin/nvcc_wrapper


