# #!/bin/bash

branch=$1

cd /lustre/orion/phy122/scratch/castia5/globus-compute/pumi-pic-test
source env.sh

# # Kokkos
# git clone -b 4.1.00 git@github.com:Kokkos/kokkos.git
# bdir=$PWD/build-kokkos
# rm -rf $bdir
# cmake -S kokkos -B $bdir \
#  -DCMAKE_BUILD_TYPE=RelWithDebInfo\
#  -DCMAKE_CXX_COMPILER=CC\
#  -DCMAKE_CXX_EXTENSIONS=OFF\
#  -DKokkos_ENABLE_TESTS=OFF\
#  -DKokkos_ENABLE_EXAMPLES=OFF\
#  -DKokkos_ENABLE_SERIAL=ON\
#  -DKokkos_ENABLE_OPENMP=OFF\
#  -DKokkos_ENABLE_HIP=ON\
#  -DKokkos_ARCH_VEGA90A=ON\
#  -DKokkos_ENABLE_DEBUG=OFF\
#  -DCMAKE_INSTALL_PREFIX=$bdir/install
# cmake --build $bdir -j8 --target install

# #Omega_h
# git clone -b scorec-v10.8.4 git@github.com:SCOREC/omega_h.git
# bdir=$PWD/build-omega_h
# rm $bdir -rf
# cmake -S omega_h -B $bdir \
#   -DCMAKE_INSTALL_PREFIX=$bdir/install \
#   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#   -DBUILD_SHARED_LIBS=OFF \
#   -DOmega_h_USE_CUDA=OFF \
#   -DOmega_h_USE_MPI=ON \
#   -DOmega_h_USE_OpenMP=OFF \
#   -DCMAKE_CXX_COMPILER=CC \
#   -DOmega_h_USE_Kokkos=ON \
#   -DOmega_h_USE_CUDA_AWARE_MPI=ON \
#   -DKokkos_PREFIX=$PWD/build-kokkos/install \
#   -DBUILD_TESTING=ON
# cmake --build $bdir -j8 --target install

# #Engpar
# git clone git@github.com:SCOREC/EnGPar.git
# bdir=$PWD/build-EnGPar
# cmake -S EnGPar -B $bdir \
#   -DCMAKE_INSTALL_PREFIX=$bdir/install \
#   -DCMAKE_C_COMPILER="mpicc" \
#   -DCMAKE_CXX_COMPILER="mpicxx" \
#   -DCMAKE_CXX_FLAGS="-std=c++11" \
#   -DENABLE_PARMETIS=OFF \
#   -DENABLE_PUMI=OFF \
#   -DIS_TESTING=OFF
# cmake --build $bdir -j8 --target install

# #Cabana
# git clone https://github.com/ECP-copa/Cabana.git cabana
# bdir=$PWD/build-cabana
# rm $bdir -rf
# cmake -S cabana -B $bdir \
#   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#   -DCMAKE_CXX_COMPILER=CC \
#   -DCMAKE_PREFIX_PATH=$PWD/build-kokkos/install \
#   -DCMAKE_INSTALL_PREFIX=$bdir/install
# cmake --build $bdir -j8 --target install

#Pumi-Pic
bdir=$PWD/build-pumi-pic
rm pumi-pic -rf
rm $bdir -rf
git clone --recursive git@github.com:SCOREC/pumi-pic.git
cd pumi-pic && git checkout $branch && cd -
cmake -S pumi-pic -B $bdir \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIS_TESTING=ON \
  -DPS_IS_TESTING=ON \
  -DPS_USE_GPU_AWARE_MPI=ON \
  -DTEST_DATA_DIR=$PWD/pumi-pic/pumipic-data \
  -DOmega_h_PREFIX=$PWD/build-omega_h/install \
  -DKokkos_PREFIX=$PWD/build-kokkos/install \
  -DEnGPar_PREFIX=$PWD/build-EnGPar/install \
  -DCabana_PREFIX=$PWD/build-cabana/install  \
  -DENABLE_CABANA=on \
  -DCMAKE_INSTALL_PREFIX=$bdir/install
cmake --build $bdir -j8 --target install