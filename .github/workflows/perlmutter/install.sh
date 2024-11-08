#!/bin/bash

branch=$1

cd $SCRATCH/globus-compute/pumi-pic-test

export root=$PWD
module load cmake

function getname() {
  name=$1
  machine=perlmutter
  buildSuffix=${machine}-cuda
  echo "build-${name}"
}
export engpar=$root/`getname engpar`/install # This is where engpar will be (or is) installed
export kk=$root/`getname kokkos`/install   # This is where kokkos will be (or is) installed
export oh=$root/`getname omegah`/install  # This is where omega_h will be (or is) installed
export cab=$root/`getname cabana`/install # This is where cabana will be (or is) installed
export pumipic=$root/`getname pumi-pic`/install # This is where PumiPIC will be (or is) installed
export CMAKE_PREFIX_PATH=$engpar:$kk:$kk/lib64/cmake:$oh:$cab:$pumipic:$CMAKE_PREFIX_PATH
export MPICH_CXX=$root/kokkos/bin/nvcc_wrapper

# #kokkos
# git clone -b 4.1.00 https://github.com/kokkos/kokkos.git
# cmake -S kokkos -B ${kk%%install} \
#   -DCMAKE_INSTALL_PREFIX=$kk \
#   -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
#   -DKokkos_ARCH_AMPERE80=ON \
#   -DKokkos_ENABLE_SERIAL=ON \
#   -DKokkos_ENABLE_OPENMP=off \
#   -DKokkos_ENABLE_CUDA=on \
#   -DKokkos_ENABLE_CUDA_LAMBDA=on \
#   -DKokkos_ENABLE_DEBUG=on
# cmake --build ${kk%%install} -j 24 --target install

# #engpar
# unset MPICH_CXX #don't want nvcc_wrapper for engpar
# git clone https://github.com/SCOREC/EnGPar.git
# cmake -S EnGPar -B ${engpar%%install} \
#   -DCMAKE_INSTALL_PREFIX=$engpar \
#   -DCMAKE_BUILD_TYPE="Release" \
#   -DCMAKE_C_COMPILER=cc \
#   -DCMAKE_CXX_COMPILER=CC \
#   -DCMAKE_CXX_FLAGS="-std=c++11" \
#   -DENABLE_PARMETIS=OFF \
#   -DENABLE_PUMI=OFF \
#   -DIS_TESTING=OFF
# cmake --build ${engpar%%install} -j 24 --target install
# export MPICH_CXX=$root/kokkos/bin/nvcc_wrapper #restore use of nvcc_wrapper

# #omegah
# git clone -b scorec-v10.8.4 https://github.com/SCOREC/omega_h.git
# cmake -S omega_h -B ${oh%%install} \
#   -DCMAKE_INSTALL_PREFIX=$oh \
#   -DCMAKE_BUILD_TYPE="Release" \
#   -DBUILD_SHARED_LIBS=OFF \
#   -DOmega_h_USE_Kokkos=ON \
#   -DOmega_h_USE_CUDA=on \
#   -DOmega_h_CUDA_ARCH=80 \
#   -DOmega_h_USE_MPI=on  \
#   -DBUILD_TESTING=off  \
#   -DCMAKE_C_COMPILER=cc \
#   -DCMAKE_CXX_COMPILER=CC \
#   -DKokkos_PREFIX=$kk/lib64/cmake
# cmake --build ${oh%%install} -j 24 --target install

# #cabana
# git clone -b 0.6.1 https://github.com/ECP-copa/Cabana.git cabana
# cmake -S cabana -B ${cab%%install} \
#   -DCMAKE_INSTALL_PREFIX=$cab \
#   -DCMAKE_BUILD_TYPE="Release" \
#   -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
#   -DCabana_ENABLE_TESTING=OFF \
#   -DCabana_ENABLE_EXAMPLES=OFF
# cmake --build ${cab%%install} -j 24 --target install

#pumipic
rm $pumipic -rf
rm pumi-pic -rf
git clone --recursive https://github.com/SCOREC/pumi-pic.git
cd pumi-pic && git checkout $branch && cd -
cmake -S pumi-pic -B ${pumipic%%install} \
  -DCMAKE_INSTALL_PREFIX=$pumipic \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=CC \
  -DENABLE_CABANA=ON \
  -DTEST_DATA_DIR=$root/pumi-pic/pumipic-data \
  -DOmega_h_PREFIX=$oh \
  -DEnGPar_PREFIX=$engpar \
  -DIS_TESTING=ON \
  -DBUILD_TESTING=ON \
  -DPS_IS_TESTING=ON
cmake --build ${pumipic%%install} -j 24 --target install
