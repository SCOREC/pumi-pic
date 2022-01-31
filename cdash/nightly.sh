#!/bin/bash -x

#load system modules
source /etc/profile.d/modules.sh
source /etc/profile

module unuse /opt/scorec/spack/lmod/linux-rhel7-x86_64/Core
module use /opt/scorec/spack/v0154_2/lmod/linux-rhel7-x86_64/Core
module load \
gcc/7.4.0 \
mpich/3.3.2 \
zoltan/3.83-int32 \
cmake/3.20.0 \
gdb \
netcdf-cxx4/4.3.1

export root=$PWD

function getname() {
  name=$1
  machine=`hostname -s`
  buildSuffix=${machine}-cuda
  echo "build-${name}-${buildSuffix}"
}
export engpar=$root/`getname engpar`/install # This is where engpar will be (or is) installed
export kk=$root/`getname kokkos`/install   # This is where kokkos will be (or is) installed
export oh=$root/`getname omegah`/install  # This is where omega_h will be (or is) installed
export cab=$root/`getname cabana`/install # This is where cabana will be (or is) installed
export pumipic=$root/`getname pumipic`/install # This is where PumiPIC will be (or is) installed
export CMAKE_PREFIX_PATH=$engpar:$kk:$kk/lib64/cmake:$oh:$cab:$pumipic:$CMAKE_PREFIX_PATH
export MPICH_CXX=$root/kokkos/bin/nvcc_wrapper

#kokkos
cd $root
#git clone -b 3.4.01 git@github.com:kokkos/kokkos.git
[ -d $kk ] && rm -rf ${kk%%install}
mkdir -p $kk
cd $_/..
cmake ../kokkos \
  -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
  -DKokkos_ARCH_TURING75=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=off \
  -DKokkos_ENABLE_CUDA=on \
  -DKokkos_ENABLE_CUDA_LAMBDA=on \
  -DKokkos_ENABLE_DEBUG=on \
  -DCMAKE_INSTALL_PREFIX=$PWD/install
make -j 24 install

##engpar
cd $root
#git clone git@github.com:SCOREC/EnGPar.git
cd EnGPar
git pull
cd -
[ -d $engpar ] && rm -rf ${engpar%%install}
mkdir -p $engpar
cd $_/..
cmake ../EnGPar \
  -DCMAKE_INSTALL_PREFIX=$engpar \
  -DCMAKE_C_COMPILER="mpicc" \
  -DCMAKE_CXX_COMPILER="mpicxx" \
  -DCMAKE_CXX_FLAGS="-std=c++11" \
  -DENABLE_PARMETIS=OFF \
  -DENABLE_PUMI=OFF \
  -DIS_TESTING=OFF
make install -j8

#omegah
cd $root
#git clone git@github.com:SCOREC/omega_h.git
cd omega_h
git pull
cd -
[ -d $oh ] && rm -rf ${oh%%install}
mkdir -p $oh
cd ${oh%%install}
cmake ../omega_h \
  -DCMAKE_INSTALL_PREFIX=$oh \
  -DBUILD_SHARED_LIBS=OFF \
  -DOmega_h_USE_Kokkos=ON \
  -DOmega_h_USE_CUDA=on \
  -DOmega_h_CUDA_ARCH=75 \
  -DOmega_h_USE_MPI=on  \
  -DBUILD_TESTING=on  \
  -DCMAKE_CXX_COMPILER=`which mpicxx` \
  -DKokkos_PREFIX=$kk/lib64/cmake
make VERBOSE=1 -j8 install
#ctest -E warp_test_parallel # see https://github.com/SCOREC/pumi-pic/pull/65#issuecomment-824335130

d=/lore/cwsmith/nightlyBuilds/pumipic
cd $d/repos/pumipic
git pull
git submodule init
git submodule update
#remove old compilation
[ -d build_pumipic ] && rm -rf build_pumipic/

#run nightly test script
ctest -VV -D Nightly -S $d/repos/pumipic/cdash/nightly.cmake
