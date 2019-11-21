module load gcc
module load mpich
module load cmake
module load netcdf
module load vim
ncxx=/lore/gopan/install/build-netcdfcxx431/install
export NETCDF_PREFIX=$ncxx
kk=/lore/gopan/install/build-kokkos-cuda-debug-profile-rhel7/install
omega_h=/lore/gopan/install/build-omegah-cuda-rhel7/install
ps=/lore/gopan/install/particle_structures_deb_prof/install
cuda=/usr/local/cuda-10.1
export PATH=$cuda/bin:$PATH
export LD_LIBRARY_PATH=$cuda/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${kk}/lib/CMake/Kokkos/:$omega_h:$ps:$ncxx:$CMAKE_PREFIX_PATH
kksrc=/lore/gopan/install/kokkos
export MPICH_CXX=${kksrc}/bin/nvcc_wrapper


